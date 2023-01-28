import argparse
import base64
import gc
import inspect
import io
import os
import shutil
import time
from typing import List
from types import SimpleNamespace

import numpy as np
import nvtx
import onnx
import tensorrt as trt
import torch
import tqdm
from cuda import cudart
import PIL
from diffusers.models import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
    DPMSolverSinglestepScheduler,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from PIL import Image
from polygraphy import cuda
from transformers import CLIPTokenizer

from schedulers import change_scheduler
from models import CLIP, VAE, UNet
from utilities import TRT_LOGGER, Engine, save_image
from core.types import Scheduler, Txt2ImgQueueEntry



def load_trt_model():
    trt_model = load_trt_model_old("CompVis/stable-diffusion-v1-4", [""], 512, 512, 50)
    return trt_model


def load_trt_model_old(model, prompt, img_height, img_width, num_inference_steps):
    #global trt_model
    #global loaded_model
    # if a model is already loaded, remove it from memory
    try:
        trt_model.teardown()
    except:
        pass

    args = SimpleNamespace()
    args.build_dynamic_shape = True


    engine_dir = f"engine/{model}"
    onnx_dir = "onnx"

    max_batch_size = 16
    if args.build_dynamic_shape:
        max_batch_size = 4

    if len(prompt) > max_batch_size:
        raise ValueError(
            f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
        )

    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    args.scheduler = "Scheduler.euler"
    args.hf_token=None
    args.verbose=False
    args.nvtx_profile=False

    # Initialize demo
    trt_model = TRTModel(
        model_path=model,
        denoising_steps=5,
        denoising_fp16=True,
        scheduler=args.scheduler,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
    )
    args.onnx_opset=16
    args.force_onnx_export=False
    args.force_onnx_optimize=False
    args.force_engine_build=False
    args.build_static_batch=False
    args.onnx_minimal_optimization=False
    args.build_dynamic_shape=True
    args.build_preview_features=True
    trt_model.loadEngines(
        engine_dir,
        onnx_dir,
        args.onnx_opset,
        opt_batch_size=len(prompt),
        opt_image_height=img_height,
        opt_image_width=img_width,
        force_export=args.force_onnx_export,
        force_optimize=args.force_onnx_optimize,
        force_build=args.force_engine_build,
        minimal_optimization=args.onnx_minimal_optimization,
        static_batch=args.build_static_batch,
        static_shape=not args.build_dynamic_shape,
        enable_preview=args.build_preview_features,
    )
    trt_model.loadModules()
    return trt_model




class TRTModel:
    """
    Application showcasing the acceleration of Stable Diffusion v1.4 pipeline using NVidia TensorRT w/ Plugins.
    """

    def __init__(
        self,
        denoising_steps,
        denoising_fp16=True,
        scheduler: str = "Scheduler.dpmpp_sde_ancestral",
        # scheduler="LMSD",
        guidance_scale=7.5,
        eta=0.0,
        device="cuda",
        hf_token: str = None,
        verbose=False,
        nvtx_profile=False,
        max_batch_size=16,
        model_path="CompVis/stable-diffusion-v1-4",
    ):
        """
        Initializes the Diffusion pipeline.
        Args:
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            denoising_fp16 (bool):
                Run the denoising loop (UNet) in fp16 precision.
                When enabled image quality will be lower but generally results in higher throughput.
            guidance_scale (float):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            max_batch_size (int):
                Max batch size for dynamic batch engines.
        """
        # Only supports single image per prompt.
        self.num_images = 1
        #self.scheduler = Scheduler.euler_a
        self.denoising_steps = denoising_steps
        self.denoising_fp16 = denoising_fp16
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile
        #self.scheduler = scheduler
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.half().to("cuda")
        # A scheduler to be used in combination with unet to denoise the encoded image latens.
        # This demo uses an adaptation of LMSDiscreteScheduler or DPMScheduler:
        sched_opts = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False,
        }

        change_scheduler(self, scheduler, config=sched_opts)

        self.tokenizer = None

        self.unet_model_key = "unet_fp16" if denoising_fp16 else "unet"
        self.models = {
            "clip": CLIP(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
            self.unet_model_key: UNet(
                model_path=model_path,
                hf_token=hf_token,
                fp16=denoising_fp16,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
            "vae": VAE(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
        }

        self.engine = {}
        self.stream = cuda.Stream()

    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def getModelPath(self, name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")

    def buildOnlyEngines(
        self,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        minimal_optimization=False,
        static_batch=False,
        static_shape=True,
        enable_preview=False,
    ):
        print("[I] Compile only mode")
        for model_name, obj in self.models.items():
            engine = Engine(model_name, engine_dir)
            onnx_path = self.getModelPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.getModelPath(model_name, onnx_dir)
            print(f"Exporting model: {onnx_path}")
            model = obj.get_model()
            with torch.inference_mode(), torch.autocast("cuda"):
                inputs = obj.get_sample_input(
                    opt_batch_size, opt_image_height, opt_image_width
                )
                torch.onnx.export(
                    model,
                    inputs,
                    onnx_path,
                    export_params=True,
                    opset_version=onnx_opset,
                    do_constant_folding=True,
                    input_names=obj.get_input_names(),
                    output_names=obj.get_output_names(),
                    dynamic_axes=obj.get_dynamic_axes(),
                )
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Generating optimizing model: {onnx_opt_path}")
            onnx_opt_graph = obj.optimize(
                onnx.load(onnx_path), minimal_optimization=minimal_optimization
            )
            onnx.save(onnx_opt_graph, onnx_opt_path)

            # Build engine
            engine.build(
                onnx_opt_path,
                fp16=True,
                input_profile=obj.get_input_profile(
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    static_batch=static_batch,
                    static_shape=static_shape,
                ),
                enable_preview=enable_preview,
            )
            engine.__del__()
            del engine
            gc.collect()
            torch.cuda.empty_cache()

    def loadEngines(
        self,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        minimal_optimization=False,
        static_batch=False,
        static_shape=True,
        enable_preview=False,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.
        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            minimal_optimization (bool):
                Apply minimal optimizations during build (no plugins).
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_preview (bool):
                Enable TensorRT preview features.
        """
        # Build engines
        for model_name, obj in self.models.items():
            engine = Engine(model_name, engine_dir)
            if force_build or not os.path.exists(engine.engine_path):
                onnx_path = self.getModelPath(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.getModelPath(model_name, onnx_dir)
                if not os.path.exists(onnx_opt_path):
                    # Export onnx
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model()
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(
                                opt_batch_size, opt_image_height, opt_image_width
                            )
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=obj.get_input_names(),
                                output_names=obj.get_output_names(),
                                dynamic_axes=obj.get_dynamic_axes(),
                            )
                        del model
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")
                    # Optimize onnx
                    if force_optimize or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = obj.optimize(
                            onnx.load(onnx_path),
                            minimal_optimization=minimal_optimization,
                        )
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")
                # Build engine
                engine.build(
                    onnx_opt_path,
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size,
                        opt_image_height,
                        opt_image_width,
                        static_batch=static_batch,
                        static_shape=static_shape,
                    ),
                    enable_preview=enable_preview,
                )
            self.engine[model_name] = engine

        # Separate iteration to activate engines
        for model_name, obj in self.models.items():
            self.engine[model_name].activate()
        gc.collect()

    def loadModules(
        self,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def txt2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        "Run text2image job"

        raise NotImplementedError
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        self.scheduler.sigmas = self.scheduler.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            self.timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            self.timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        schedule_timesteps = self.timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = self.scheduler.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples
    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        guidance_scale=1.0,
        eta=0.0,
        warmup=False,
        verbose=False,
        seed=None,
        output_dir="static/output",
        num_of_infer_steps=50,
        scheduler: str = "Scheduler.euler_a",
        init_image=None
    ):
        """
        Run the diffusion pipeline.
        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Enable verbose logging.
        """
        # Process inputs
        batch_size = len(prompt)
        assert len(prompt) == len(negative_prompt)

        ## Number of infer steps
        self.denoising_steps = num_of_infer_steps

        sched_opts = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False,
        }

        change_scheduler(self, scheduler, sched_opts)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        # Create profiling events
        events = {}
        for stage in ["clip", "denoise", "vae"]:
            for marker in ["start", "stop"]:
                events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width),
                device=self.device,
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run Stable Diffusion pipeline
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # latents need to be generated on the target device
            unet_channels = 4  # unet.in_channels
            latents_shape = (
                batch_size * self.num_images,
                unet_channels,
                latent_height,
                latent_width,
            )
            latents_dtype = torch.float32  # text_embeddings.dtype
            random = torch.randn(
                latents_shape,
                device=self.device,
                dtype=latents_dtype,
                generator=generator,
            )
            image_bytes = base64.b64decode(init_image)
            buf = io.BytesIO(image_bytes)


            #print(type(image_bytes))
            init_image = PIL.Image.open(buf)
            init_image = init_image.convert('RGB').resize((512, 512), resample=PIL.Image.LANCZOS)
            #init_image.save("testcam.png")
            #sample = np.array(init_image)
            #sample = ((sample.astype(float) / 255.0) * 2) - 1
            ##latents = sample[None].transpose(0, 3, 1, 2).astype(np.float32)
            #latents = torch.from_numpy(sample).to("cuda")
            #print(latents.shape)
            self.scheduler.set_timesteps(self.denoising_steps, device=self.device)
            #print(self.models["vae"].__dir__)

            #latents = self.models["vae"].encode(latents)
            #sigmas = self.scheduler.sigmas
            #sigmas = sigmas.to(latents_dtype)
            #latents = latents * sigmas[0]
            #self.scheduler.timesteps = timesteps

            # Scale the initial noise by the standard deviation required by the scheduler
            #latents = latents * self.scheduler.init_noise_sigma



            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            if self.nvtx_profile:
                nvtx_clip = nvtx.start_range(message="clip", color="green")
            cudart.cudaEventRecord(events["clip-start"], 0)

            # Tokenize input
            text_input_ids = (
                self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                .input_ids.type(torch.int32)
                .to(self.device)
            )

            # CLIP text encoder
            text_input_ids_inp = cuda.DeviceView(
                ptr=text_input_ids.data_ptr(),
                shape=text_input_ids.shape,
                dtype=np.int32,
            )
            text_embeddings = self.runEngine("clip", {"input_ids": text_input_ids_inp})[
                "text_embeddings"
            ]

            # Duplicate text embeddings for each generation per prompt
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, self.num_images, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * self.num_images, seq_len, -1
            )

            do_classifier_free_guidance = guidance_scale > 1.0

            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                max_length = text_input_ids.shape[-1]
                uncond_input_ids = (
                    self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    .input_ids.type(torch.int32)
                    .to(self.device)
                )
                uncond_input_ids_inp = cuda.DeviceView(
                    ptr=uncond_input_ids.data_ptr(),
                    shape=uncond_input_ids.shape,
                    dtype=np.int32,
                )
                uncond_embeddings = self.runEngine(
                    "clip", {"input_ids": uncond_input_ids_inp}
                )["text_embeddings"]

                # Duplicate unconditional embeddings for each generation per prompt
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(1, self.num_images, 1)
                uncond_embeddings = uncond_embeddings.view(
                    batch_size * self.num_images, seq_len, -1
                )

                # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            if self.denoising_fp16:
                text_embeddings = text_embeddings.to(dtype=torch.float16)

            cudart.cudaEventRecord(events["clip-stop"], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_clip)


            #timesteps = self.scheduler.timesteps
            #init_timestep = self.scheduler.timesteps[int(self.denoising_steps * 0.75)]
            #timesteps_to_iterate_over = self.scheduler.timesteps[int(self.denoising_steps * 0.75):]
            #latents = latents * self.scheduler.init_noise_sigma
            #latents = self.scheduler.add_noise(latents, init_timestep, timesteps_to_iterate_over)
            #self.scheduler.timesteps = timesteps_to_iterate_over

            #extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)

            cudart.cudaEventRecord(events["denoise-start"], 0)
            sample = preprocess(init_image).to("cuda")

            latents = self.vae.encode(sample).latent_dist.sample(generator=generator)
            timesteps, latents, t_start, extra_step_kwargs = self.prepare_timesteps(self.denoising_steps, 0.90,
                                                                                    generator, latents)
            #self.scheduler.timesteps = timesteps

            print(latents.shape)
            #latents = self.runEngine("vae", {"latents": latents})

            for step_index, timestep in enumerate(tqdm.tqdm(timesteps)):
                if self.nvtx_profile:
                    nvtx_latent_scale = nvtx.start_range(
                        message="latent_scale", color="pink"
                    )
                # expand the latents if we are doing classifier free guidance
                
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                #LMSDiscreteScheduler.scale_model_input()
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_scale)

                # predict the noise residual
                if self.nvtx_profile:
                    nvtx_unet = nvtx.start_range(message="unet", color="blue")
                dtype = np.float16 if self.denoising_fp16 else np.float32
                if timestep.dtype != torch.float32:
                    timestep_float = timestep.float()
                else:
                    timestep_float = timestep
                sample_inp = cuda.DeviceView(
                    ptr=latent_model_input.data_ptr(),
                    shape=latent_model_input.shape,
                    dtype=np.float32,
                )
                timestep_inp = cuda.DeviceView(
                    ptr=timestep_float.data_ptr(),
                    shape=timestep_float.shape,
                    dtype=np.float32,
                )
                embeddings_inp = cuda.DeviceView(
                    ptr=text_embeddings.data_ptr(),
                    shape=text_embeddings.shape,
                    dtype=dtype,
                )
                noise_pred = self.runEngine(
                    self.unet_model_key,
                    {
                        "sample": sample_inp,
                        "timestep": timestep_inp,
                        "encoder_hidden_states": embeddings_inp,
                    },
                )["latent"]
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_unet)

                if self.nvtx_profile:
                    nvtx_latent_step = nvtx.start_range(
                        message="latent_step", color="pink"
                    )
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs
                ).prev_sample

                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_step)

            latents = 1.0 / 0.18215 * latents
            cudart.cudaEventRecord(events["denoise-stop"], 0)

            if self.nvtx_profile:
                nvtx_vae = nvtx.start_range(message="vae", color="red")
            cudart.cudaEventRecord(events["vae-start"], 0)
            sample_inp = cuda.DeviceView(
                ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
            )
            images = self.runEngine("vae", {"latent": sample_inp})["images"]
            #images = self.vae.decode(sample_inp)
            cudart.cudaEventRecord(events["vae-stop"], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_vae)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()
            if not warmup:
                print("|------------|--------------|")
                print("| {:^10} | {:^12} |".format("Module", "Latency"))
                print("|------------|--------------|")
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "CLIP",
                        cudart.cudaEventElapsedTime(
                            events["clip-start"], events["clip-stop"]
                        )[1],
                    )
                )
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "UNet x " + str(self.denoising_steps),
                        cudart.cudaEventElapsedTime(
                            events["denoise-start"], events["denoise-stop"]
                        )[1],
                    )
                )
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "VAE",
                        cudart.cudaEventElapsedTime(
                            events["vae-start"], events["vae-stop"]
                        )[1],
                    )
                )
                print("|------------|--------------|")
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "Pipeline", (e2e_toc - e2e_tic) * 1000.0
                    )
                )
                print("|------------|--------------|")

                # Save image
                image_name_prefix = (
                    "sd-"
                    + ("fp16" if self.denoising_fp16 else "fp32")
                    + "".join(
                        set(
                            [
                                "-" + prompt[i].replace(" ", "_")[:10]
                                for i in range(batch_size)
                            ]
                        )
                    )
                    + "-"
                )

                #imgs = save_image(images, output_dir, image_name_prefix)
            return str(e2e_toc - e2e_tic), images

    def prepare_timesteps(self, num_inference_steps, strength, generator, init_latent, eta=0, batch_size=1):
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        print(f"Init timestep", init_timestep, "offset", offset)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size * 1, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latent.shape, generator=generator, device=self.device, dtype=torch.float32)
        init_latent = 0.18215 * init_latent
        init_latent = self.scheduler.add_noise(init_latent, noise, timesteps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        latents = init_latent

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        print("t-start", t_start)

        return timesteps, latents, t_start, extra_step_kwargs
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.flip(image, 2).copy()
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0