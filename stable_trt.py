#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
from typing import List, Optional, Union
from PIL import Image
from cuda import cudart
from models import CLIP, UNet, VAE
import numpy as np
import nvtx
import os
import onnx
from polygraphy import cuda
import time
import tqdm
import torch
from transformers import CLIPTokenizer
import tensorrt as trt
from utilities_pix import Engine, TRT_LOGGER
import gc
from diffusers import AutoencoderKL
from torchvision import transforms
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
import PIL
from typing import TypeVar


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


SchedulerType = TypeVar(
    "SchedulerType",
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)


class DemoDiffusion:
    """
    Application showcasing the acceleration of Stable Diffusion v1.4 pipeline using NVidia TensorRT w/ Plugins.
    """

    def __init__(
        self,
        denoising_steps: int,
        denoising_fp16: bool = True,
        scheduler: SchedulerType = EulerAncestralDiscreteScheduler,
        guidance_scale: int = 7.5,
        device: Union[str, torch.device] = "cuda",
        output_dir: str = ".",
        hf_token: str = None,
        verbose: bool = False,
        nvtx_profile: bool = False,
        max_batch_size: int = 16,
        model_path: str = "/app/volta-trt/proto-eclipse",
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

        self.denoising_steps = denoising_steps
        self.denoising_fp16 = denoising_fp16
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        # A scheduler to be used in combination with unet to denoise the encoded image latens.
        # This demo uses an adaptation of LMSDiscreteScheduler or DPMScheduler:
        sched_opts = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        }
        sched_opts = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            # "clip_sample": False,
            "num_train_timesteps": 1000,
            # "prediction_type": "epsilon",
            # "set_alpha_to_one": False,
            # "skip_prk_steps": True,
            # "steps_offset": 1,
            # "trained_betas": None
        }
        self.scheduler = EulerDiscreteScheduler(**sched_opts)
        self.tokenizer = None

        self.unet_model_key = "unet_fp16" if denoising_fp16 else "unet"
        self.models = {
            "clip": CLIP(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
                model_path=model_path,
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
                model_path=model_path,
            ),
        }
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16
        )
        self.vae = self.vae.to("cuda")
        self.engine = {}
        self.stream = cuda.Stream()
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def getModelPath(self, name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")

    def encode_image(self, image, generator, width, height):
        if not image.size == (width, height):
            image = image.resize((width, height), resample=Image.LANCZOS)
        init_image = preprocess(image)
        init_image = init_image.to(device=self.device, dtype=self.vae.dtype)
        init_latent_dist = self.vae.encode(init_image.to(self.vae.dtype)).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        return init_latents.to(torch.float32)

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
            model = obj.get_model(self.model_path)
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
        self.scheduler.set_timesteps(self.denoising_steps)
        # Pre-compute latent input scales and linear multistep coefficients
        # self.scheduler.configure()

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def prepare_timesteps(
        self, num_inference_steps, strength, generator, init_latent, eta=0, batch_size=1
    ):
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        print(f"Init timestep", init_timestep, "offset", offset)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size * 1, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(
            init_latent.shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        init_latent = 0.18215 * init_latent
        init_latent = self.scheduler.add_noise(init_latent, noise, timesteps)

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

        latents = init_latent

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        print("t-start", t_start)

        return timesteps, latents, t_start, extra_step_kwargs

    def infer(
        self,
        prompt: List[str],
        negative_prompt: List[str],
        height: int,
        width: int,
        guidance_scale: float = 7.5,
        warmup: bool = False,
        verbose: bool = False,
        seed: int = None,
        output_dir: str = "static/output",
        num_inference_steps: int = 50,
        init_image: Image.Image = None,
        init_strength: float = None,
        eta: Optional[float] = 0,
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
        generator = torch.Generator(self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Spatial dimensions of latent tensor
        latent_height = height // 8
        latent_width = width // 8
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        unet_channels = 4  # unet.in_channels

        if init_image is not None and init_strength is not None:
            with torch.inference_mode():
                init_latents = self.encode_image(
                    init_image, generator, width, height
                ).to(dtype=torch.float32, device=self.device)
                timesteps, latents, t_start, extra_kwargs = self.prepare_timesteps(
                    num_inference_steps,
                    init_strength,
                    generator,
                    init_latents,
                    eta,
                    batch_size,
                )
                #print("timesteps", timesteps)
        else:
            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
                TRT_LOGGER
            ) as runtime:
                # latents need to be generated on the target device
                latents_shape = (
                    batch_size * self.num_images,
                    unet_channels,
                    latent_height,
                    latent_width,
                )
                latents_dtype = torch.float32  # text_embeddings.dtype
                init_latents = torch.randn(
                    latents_shape,
                    device=self.device,
                    dtype=latents_dtype,
                    generator=generator,
                )
                timesteps, latents, t_start, extra_kwargs = self.prepare_timesteps(
                    num_inference_steps,
                    init_strength,
                    generator,
                    init_latents,
                    eta,
                    batch_size,
                )

        # Create profiling events
        events = {}
        for stage in ["clip", "denoise", "vae"]:
            for marker in ["start", "stop"]:
                events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, height, width),
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
            # Scale the initial noise by the standard deviation required by the scheduler

            # latents = latents * self.scheduler.init_noise_sigma

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

            max_length = text_input_ids.shape[-1]
            uncond_input_ids = (
                self.tokenizer(
                    negative_prompt,
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

            cudart.cudaEventRecord(events["denoise-start"], 0)
            for step_index, timestep in enumerate(tqdm.tqdm(timesteps)):
                #print("timestep shape", timestep)
                if self.nvtx_profile:
                    nvtx_latent_scale = nvtx.start_range(
                        message="latent_scale", color="pink"
                    )
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                # LMSDiscreteScheduler.scale_model_input()
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
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                latents = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_kwargs
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
            cudart.cudaEventRecord(events["vae-stop"], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_vae)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()
            """if not warmup:
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
                # image_name_prefix = 'sd-'+('fp16' if self.denoising_fp16 else 'fp32')+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(batch_size)]))+'-'
                # images = save_image(images, output_dir, image_name_prefix)"""
            return str(e2e_toc - e2e_tic), images


def load_trt(
    model,
    scheduler,
    batch_size=1,
    img_height=512,
    img_width=512,
    num_inference_steps=50,
) -> DemoDiffusion:
    global trt_model
    global loaded_model
    # if a model is already loaded, remove it from memory
    try:
        trt_model.teardown()
    except:
        pass

    engine_dir = f"engine/{model}"
    onnx_dir = "onnx"

    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # Initialize demo
    trt_model = DemoDiffusion(
        model_path=model,
        denoising_steps=num_inference_steps,
        denoising_fp16=True,
        scheduler=scheduler,
        hf_token="",
        verbose=True,
        nvtx_profile=False,
        max_batch_size=16,
    )

    trt_model.loadEngines(
        engine_dir,
        onnx_dir,
        16,
        opt_batch_size=batch_size,
        opt_image_height=img_height,
        opt_image_width=img_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        minimal_optimization=False,
        static_batch=False,
        static_shape=False,
        enable_preview=True,
    )
    trt_model.loadModules()
    loaded_model = model
    return trt_model
