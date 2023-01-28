#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import argparse

import PIL
from cuda import cudart
from diffusers import AutoencoderKL
from diffusers.utils import deprecate

from core.diffusers import DPMSolverSinglestepScheduler
from core.diffusers.utils import randn_tensor
from models import CLIP, UNet, VAE
import numpy as np
import nvtx
import os
import random
from polygraphy import cuda
import torch
from transformers import CLIPTokenizer
import uuid
import tensorrt as trt
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from utilities_pix import Engine
from pytorch_model import inference, load_model
import gc
from PIL import Image
import inspect
import tqdm

from typing import List
def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Demo")
    # Stable Diffusion configuration
    parser.add_argument('--prompt', nargs = '*', help="Text prompt(s) to guide image generation")
    parser.add_argument('--negative-prompt', nargs = '*', default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--repeat-prompt', type=int, default=1, choices=[1, 2, 4, 8, 16], help="Number of times to repeat the prompt (batch size multiplier)")
    parser.add_argument('--height', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=512, help="Height of image to generate (must be multiple of 8)")
    # parser.add_argument('--num-images', type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument('--denoising-steps', type=int, default=50, help="Number of denoising steps")
    parser.add_argument('--denoising-prec', type=str, default='fp16', choices=['fp32', 'fp16'], help="Denoiser model precision")
    parser.add_argument('--scheduler', type=str, default="DDIMScheduler", help="Scheduler for diffusion process")

    # ONNX export
    parser.add_argument('--onnx-opset', type=int, default=16, choices=range(7,18), help="Select ONNX opset version to target for exported models")
    parser.add_argument('--onnx-dir', default='onnx', help="Output directory for ONNX export")
    parser.add_argument('--force-onnx-export', action='store_true', help="Force ONNX export of CLIP, UNET, and VAE models")
    parser.add_argument('--force-onnx-optimize', action='store_true', help="Force ONNX optimizations for CLIP, UNET, and VAE models")
    parser.add_argument('--onnx-minimal-optimization', action='store_true', help="Restrict ONNX optimization to const folding and shape inference.")

    # TensorRT engine build
    parser.add_argument('--model-path', default="CompVis/stable-diffusion-v1-4", help="HuggingFace Model path")
    parser.add_argument('--engine-dir', default='engine', help="Output directory for TensorRT engines")
    parser.add_argument('--force-engine-build', action='store_true', help="Force rebuilding the TensorRT engine")
    parser.add_argument('--build-static-batch', action='store_true', help="Build TensorRT engines with fixed batch size.")
    parser.add_argument('--build-dynamic-shape', action='store_false', help="Build TensorRT engines with dynamic image shapes.")
    parser.add_argument('--build-preview-features', action='store_true', help="Build TensorRT engines with preview features.")

    # TensorRT inference
    parser.add_argument('--num-warmup-runs', type=int, default=0, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--nvtx-profile', action='store_true', help="Enable NVTX markers for performance profiling")
    parser.add_argument('--seed', type=int, default=None, help="Seed for random generator to get consistent results")

    parser.add_argument('--output-dir', default='output', help="Output directory for logs and image artifacts")
    parser.add_argument('--hf-token', type=str, help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    parser.add_argument('--backend', default='PT', help="PT(PyTorch) or TRT(TensorRT)")

    return parser.parse_args()

class DemoDiffusion:
    """
    Application showcasing the acceleration of Stable Diffusion v1.4 pipeline using NVidia TensorRT w/ Plugins.
    """

    def __init__(self, model_path):
  
        # Only supports single image per prompt.
        self.num_images = 1

        self.engine_dir = os.path.join("engine", model_path)
        self.scheduler_config_path = os.path.join(model_path, "scheduler")
        self.device = "cuda"

        self.tokenizer = None
        self.schedulers = {}
        self.vae = AutoencoderKL.from_pretrained(model_path,
                                                torch_dtype=torch.float32,
                                                #revision="fp32",
                                                subfolder="vae").to("cuda")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path,
                                                torch_dtype=torch.float16,
                                                revision="fp16",
                                                subfolder="text_encoder").to("cuda")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        batch_size = 1
        text_maxlen = 77
        embedding_dim = 768
        image_height = 512
        image_width = 512
        latent_height = image_height // 8
        latent_width = image_width // 8
        self.models = {
            "clip": {
                "input_ids": (batch_size, text_maxlen),
                "text_embeddings": (batch_size, text_maxlen, embedding_dim),
            },
            "unet_fp16": {
                "sample": (2, 8, 64, 64),
                "encoder_hidden_states": (
                    2 * batch_size,
                    text_maxlen,
                    embedding_dim,
                ),
                "latent": (3 * batch_size, 8, latent_height, latent_width),
            },
            "vae": {
                "latent": (batch_size, 4, latent_height, latent_width),
                "images": (batch_size, 3, image_height, image_width),
            },
        }

        self.engine = {}
        self.stream = cuda.Stream()

        self.loadEngines(self.engine_dir)
        self.loadModules()

        # Allocate buffers for TensorRT engine bindings
        for model_name, shape_dict in self.models.items():
            # print(model_name)
            self.engine[model_name].allocate_buffers(
                shape_dict=shape_dict, device=self.device
            )


    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def loadEngines(self, engine_dir):
        """
        Load engines for TensorRT accelerated inference.
        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
        """
        # Build engines
        for model_name in self.models:
            engine = Engine(model_name, engine_dir)
            self.engine[model_name] = engine
            self.engine[model_name].activate()
        gc.collect()

    def loadModules(
        self,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        schedulers = [
            "DDIMScheduler",
            "DPMSolverMultistepScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "LMSDiscreteScheduler",
            "PNDMScheduler",
        ]
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        scheduler = ["KarrasDiffusionSchedulers"]
        for scheduler in schedulers:
            self.schedulers[scheduler] = eval(scheduler).from_config(
                sched_opts
            )

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

    def infer(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta: float = 0.0,
        seed=42,

    ):

        # Process inputs
        batch_size = len(prompt)
        config = {
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

        self.scheduler = DPMSolverSinglestepScheduler.from_config(config)
        #self.scheduler = self.schedulers[scheduler]
        print(self.scheduler)
        callback_steps = 1
        #0
        self.check_inputs(prompt, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0
        #1
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")
        #2
        text_embeddings = self._encode_prompt(
            prompt, "cuda", num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        image = PIL.Image.open("result.png")
        image = preprocess(image)
        height, width = image.shape[-2:]
        self.scheduler.set_timesteps(num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run Stable Diffusion pipeline
        with torch.inference_mode():
            # latents need to be generated on the target device
            unet_channels = 8  # unet.in_channels
            latents_shape = (
                1,
                8,
                64,
                64,
            )
            latents_dtype = torch.float32  # text_embeddings.dtype
            latents = torch.randn(
                latents_shape,
                device=self.device,
                dtype=latents_dtype,
                generator=generator,
            )

            # Scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            torch.cuda.synchronize()

            # Tokenize input
            text_input_ids = (
                self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
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
            #text_embeddings = self.runEngine("clip", {"input_ids": text_input_ids_inp})[
            #    "text_embeddings"
            #]

            # Duplicate text embeddings for each generation per prompt
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            if do_classifier_free_guidance:
                uncond_tokens = List[str]
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
                uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
                uncond_embeddings = uncond_embeddings.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

                # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            text_embeddings = text_embeddings.to(dtype=torch.float16)
            # 5.
            image_latents = self.prepare_image_latents(
                image,
                batch_size,
                num_images_per_prompt,
                text_embeddings.dtype,
                "cuda",
                do_classifier_free_guidance,
                generator,
            )
            # 6. Prepare latent variables
            num_channels_latents = self.vae.config.latent_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                "cuda",
                generator,
                latents,
            )

            # 7. Check that shapes of latents and image match the UNet channels
            num_channels_image = image_latents.shape[1]
            #if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            #    raise ValueError(
            #        f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
            #        f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
            #        f" `num_channels_image`: {num_channels_image} "
            #        f" = {num_channels_latents + num_channels_image}. Please verify the config of"
            #        " `pipeline.unet` or your `image` input."
            #    )

            # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            for step_index, timestep in enumerate(tqdm.tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                #do_classifier_free_guidance = False
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                # LMSDiscreteScheduler.scale_model_input()
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                #latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                dtype = np.float16
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
                    "unet_fp16",
                    {
                        "sample": sample_inp,
                        "timestep": timestep_inp,
                        "encoder_hidden_states": embeddings_inp,
                    },
                )["latent"]
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == timestep).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                latents = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs
                ).prev_sample

            latents = 1.0 / 0.18215 * latents

            sample_inp = cuda.DeviceView(
                ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
            )
            image_old_vae = self.runEngine("vae", {"latent": sample_inp})["images"]
            print(image_old_vae)
            # 10. Post-processing
            print(latents.shape)
            print(type(latents))
            images = self.decode_latents(latents)

            """images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")"""
            print(images)
            return images
    def decode_latents(self, latents):
        #latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image



    def check_inputs(self, prompt, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            #logger.warning(
            #    "The following part of your input was truncated because CLIP can only handle sequences up to"
            #    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #)

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
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
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([text_embeddings, uncond_embeddings, uncond_embeddings])

        return text_embeddings
    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        device = "cuda"
        image = image.to(device=device, dtype=torch.float32)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


def infer_trt(saving_path, model, prompt, neg_prompt, img_height, img_width, num_inference_steps, guidance_scale, num_images_per_prompt, seed=None, scheduler='DDIMScheduler'):
    global trt_model
    global loaded_model
    print("[I] Initializing StableDiffusion demo with TensorRT Plugins")
    args = parseArgs()

    args.output_dir=saving_path
    args.prompt=[prompt]
    args.model_path=model
    args.height=img_height
    args.width=img_width
    args.repeat_prompt=num_images_per_prompt
    args.denoising_steps=num_inference_steps
    args.seed=seed
    args.guidance_scale=guidance_scale
    args.negative_prompt=[neg_prompt]
    
    print('Seed :', args.seed)
    
    args.engine_dir = os.path.join(args.engine_dir, args.model_path)

    # Process prompt
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    # print('String :', args.prompt, type(args.prompt))
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}")
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")
    
    eta = 0.0

    trt_model = DemoDiffusion(
                model_path=args.model_path
            )    
    
    images = trt_model.infer(
                prompt,
                image_height,
                image_width,
                args.denoising_steps,
                args.guidance_scale,
                negative_prompt,
                num_images_per_prompt,
                eta,
                seed
            ) 
    for i in range(images.shape[0]):
        image_path  = os.path.join("output/", "out_"+str(i+1)+'-'+str(random.randint(1000,9999))+'.png')
        print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
        Image.fromarray(images[i]).save(image_path)

    return images

    
def infer_pt(saving_path, model_path, prompt, negative_prompt, img_height, img_width, num_inference_steps, guidance_scale, num_images_per_prompt, seed):
    
    print("[+] Loading the model")
    model = load_model(model_path)
    print("[+] Model loaded")

    print("[+] Generating images...")
    # PIL images
    images, time = inference(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        img_height=img_height,
        img_width=img_width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        seed=seed,
        return_time=True,
    )
    model = None
    print("[+] Time needed to generate the images: {} seconds".format(time))

    # Save PIL images with a random name
    for img in images:
        img.save("{}/{}.png".format(saving_path, uuid.uuid4()))

    print(
        "[+] Images saved in the following path: {}".format(saving_path)
    )
    del model
    gc.collect()
    return str(round(time,2))

                
if __name__ == "__main__":

    print("[I] Initializing StableDiffusion demo with TensorRT Plugins")
    args = parseArgs()
    
    if "trt" in args.backend.lower():
        infer_trt(saving_path=args.output_dir,
                  model=args.model_path,
                  prompt=args.prompt[0],
                  neg_prompt=args.negative_prompt[0],
                  img_height=args.height,
                  img_width=args.width, 
                  num_inference_steps=args.denoising_steps,
                  guidance_scale=15,
                  num_images_per_prompt=args.repeat_prompt,
                  seed=args.seed,
                  scheduler=args.scheduler)

    if "pt" in args.backend.lower():
        infer_pt(saving_path=args.output_dir,
                  model_path=args.model_path,
                  prompt=args.prompt,
                  negative_prompt=args.negative_prompt,
                  img_height=args.height,
                  img_width=args.width, 
                  num_inference_steps=args.denoising_steps,
                  guidance_scale=12,
                  num_images_per_prompt=args.repeat_prompt,
                  seed=args.seed)


