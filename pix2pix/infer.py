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

import PIL
import inspect
from typing import List
import os

import numpy as np
from cuda import cudart
from diffusers import DEISMultistepScheduler
from polygraphy import cuda


import torch
from transformers import CLIPTokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL
from utilities import Engine

import gc

PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

class DemoDiffusion:
    """
    Application showcasing the acceleration of Stable Diffusion v1.4 pipeline using NVidia TensorRT w/ Plugins.
    """

    def __init__(self, model_path):
  
        # Only supports single image per prompt.
        self.num_images = 1

        self.engine_dir = os.path.join(model_path)
        self.scheduler_config_path = os.path.join(self.engine_dir)
        self.device = "cuda"

        self.tokenizer = None
        self.schedulers = {}

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
                "sample": (3 * batch_size, 8, latent_height, latent_width),
                "encoder_hidden_states": (
                    3 * batch_size,
                    text_maxlen,
                    embedding_dim,
                ),
                "latent": (3 * batch_size, 4, latent_height, latent_width),
            },
            "vae": {
                "latent": (batch_size, 4, latent_height, latent_width),
                "images": (batch_size, 3, image_height, image_width),
            },
            #"vae_encoder":{
            #    "image": (batch_size, 3, image_height, image_width),
            #    "image_latents": (batch_size, 4, latent_height, latent_width)
        #}
        }

        self.engine = {}
        self.stream = cuda.Stream()
        print(self.engine_dir)
        self.loadEngines(self.engine_dir)
        self.loadModules()

        self.vae_encoder = AutoencoderKL.from_pretrained("timbrooks/instruct-pix2pix",subfolder="vae").to("cuda")

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
            print(model_name, engine_dir)
            engine = Engine(model_name, engine_dir)
            self.engine[model_name] = engine
            self.engine[model_name].activate()
        gc.collect()

    def loadModules(
        self,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder="tokenizer")
        self.scheduler = DEISMultistepScheduler.from_config(
                self.scheduler_config_path
            )
        self.scheduler.config.algorithm_type = "deis"

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def prepare_extra_step_kwargs(self, generator, eta):
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
        return extra_step_kwargs

    def infer(
            self,
            prompt,
            image,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            negative_prompt=None,
            num_images_per_prompt=1,
            eta: float = 0.0,
            seed=42,
            # scheduler="DPMSolverMultistepScheduler"
    ):

        # Process inputs
        batch_size = len(prompt)
        # Create profiling events
        events = {}
        for stage in ["denoise"]:
            for marker in ["start", "stop"]:
                events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

        cudart.cudaEventRecord(events["denoise-start"], 0)
        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # self.scheduler = self.schedulers[scheduler]

        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run Stable Diffusion pipeline
        with torch.inference_mode():
            # latents need to be generated on the target device
            unet_channels = 4  # unet.in_channels
            latents_shape = (
                batch_size * num_images_per_prompt,
                unet_channels,
                image.size[1] // 8,
                image.size[0] // 8,
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
            text_embeddings = self.runEngine("clip", {"input_ids": text_input_ids_inp})[
                "text_embeddings"
            ]

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
                #elif batch_size != len(negative_prompt):
                #    raise ValueError(
                #        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                #        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                #        " the batch size of `prompt`."
                #    )
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
                # import ipdb; ipdb.set_trace()
                # Duplicate unconditional embeddings for each generation per prompt
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
                uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
                # import ipdb; ipdb.set_trace()
                # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                # import ipdb; ipdb.set_trace()
                text_embeddings = torch.cat([text_embeddings, uncond_embeddings, uncond_embeddings])

            text_embeddings = text_embeddings.to(dtype=torch.float16)

            image = preprocess(image)
            height, width = image.shape[-2:]

            image = image.to(device=self.device)

            ### VAE Encoder in PyTorch
            image_latents = self.vae_encoder.encode(image).latent_dist.mode()
            image_latents = torch.cat([image_latents], dim=0)
            ###

            ### VAE Encoder in TRT
            # image = cuda.DeviceView(
            #         ptr=image.data_ptr(),
            #         shape=image.shape,
            #         dtype=np.float16,
            #     )
            # batch_size = batch_size * num_images_per_prompt

            # image_latents = self.runEngine(
            #         "vae_encoder", {"image": image}
            #     )["image_latents"]
            # image_latents = torch.cat([image_latents], dim=0)
            ###

            if do_classifier_free_guidance:
                uncond_image_latents = torch.zeros_like(image_latents)
                image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

            # image_latents = image_latents.to(dtype=torch.float16)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 3) if do_classifier_free_guidance else latents
            )
            # LMSDiscreteScheduler.scale_model_input()
            scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps[0])
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

            dtype = np.float16
            if timesteps[0].dtype != torch.float32:
                timestep_float = timesteps[0].float()
            else:
                timestep_float = timesteps[0]
            sample_inp = cuda.DeviceView(
                ptr=scaled_latent_model_input.data_ptr(),
                shape=scaled_latent_model_input.shape,
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
            # import ipdb; ipdb.set_trace()
            if scheduler_is_in_sigma_space:
                step_index = (self.scheduler.timesteps == timesteps[0]).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]
                noise_pred = latent_model_input - sigma * noise_pred

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            if scheduler_is_in_sigma_space:
                noise_pred = (noise_pred - latents) / (-sigma)

            latents = self.scheduler.step(
                noise_pred, timesteps[0], latents, **extra_step_kwargs
            ).prev_sample
            latent_chunks = latents.chunk(16)
            result_chunks = []



            for latents in latent_chunks:
                timesteps = timesteps[1:]
                for step_index, timestep in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                    )
                    # LMSDiscreteScheduler.scale_model_input()
                    scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
                    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                    dtype = np.float16
                    if timestep.dtype != torch.float32:
                        timestep_float = timestep.float()
                    else:
                        timestep_float = timestep
                    sample_inp = cuda.DeviceView(
                        ptr=scaled_latent_model_input.data_ptr(),
                        shape=scaled_latent_model_input.shape,
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
                    # import ipdb; ipdb.set_trace()
                    if scheduler_is_in_sigma_space:
                        step_index = (self.scheduler.timesteps == timestep).nonzero().item()
                        sigma = self.scheduler.sigmas[step_index]
                        noise_pred = latent_model_input - sigma * noise_pred

                    # Perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                                noise_pred_uncond
                                + guidance_scale * (noise_pred_text - noise_pred_image)
                                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )

                    if scheduler_is_in_sigma_space:
                        noise_pred = (noise_pred - latents) / (-sigma)

                    latents = self.scheduler.step(
                        noise_pred, timestep, latents, **extra_step_kwargs
                    ).prev_sample
                result_chunks.append(latents)
            latents = torch.cat(result_chunks, dim=0)

            latents = 1 / 0.18215 * latents

            sample_inp = cuda.DeviceView(
                ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
            )
            images = self.runEngine("vae", {"latent": sample_inp})["images"]

            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            image_path = "output.png"
            pil_image = PIL.Image.fromarray(images[0])
            cudart.cudaEventRecord(events["denoise-stop"], 0)
            print(
                "| {:^10} | {:>9.2f} ms |".format(
                    "COMPLETE PIPELINE",
                    cudart.cudaEventElapsedTime(
                        events["denoise-start"], events["denoise-stop"]
                    )[1],
                )
            )

            return 0, pil_image