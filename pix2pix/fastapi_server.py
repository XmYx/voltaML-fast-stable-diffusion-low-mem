import asyncio
import base64
import io
import subprocess

import PIL
try:
    import fastapi
except:
    cmd = ["pip", "install", "fastapi", "msgpack", "diffusers", "ray", "--upgrade"]
    subprocess.run(cmd)
    cmd = ["pip", "install", "torchvision==0.13.0+cu116", "--extra-index-url", "https://download.pytorch.org/whl/cu116"]
    subprocess.run(cmd)
from typing import List, Optional, Union
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler
from fastapi import FastAPI
import torch

#from schedulers import change_scheduler
#from trt_cleanup import load_trt_model
from infer import DemoDiffusion
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from msgpack import packb



class TRTWorker:
    def __init__(self):
        self.trt = DemoDiffusion(
                    model_path="engine/pix2pix/"
                )

    def generate(self, body, init_image):

        dur, result = self.trt.infer(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            height=body.height,
            width=body.width,
            guidance_scale=body.guidance_scale,
            seed=body.seed,
            num_inference_steps=body.num_inference_steps,
            image=init_image,
            image_guidance_scale=body.strength,
            eta=0.0,
        )
        return result

# in main process
global worker
def on_startup():
   global worker
   worker = TRTWorker()




def image_tobytes(img: Image.Image):
    if img is None:
        print("no_image_returned")
        return None
    if type(img) == bytes:
        return img
    b = io.BytesIO()
    b.seek(0)
    img.save(b, format="PNG")
    b.seek(0)
    return b.getvalue()


class InferenceArgs(BaseModel):
    prompt: Optional[List[str]] = [
        "Portrait of a gorgeous princess, by WLOP, Stanley Artgerm Lau, trending on ArtStation"]
    negative_prompt: Optional[List[str]] = [
        "Horrible, very ugly, jpeg artifacts, messy, warped, split, bad anatomy, malformed body, malformed, warped, fake, 3d, drawn, hideous, disgusting"]
    height: Optional[int] = 512
    width: Optional[int] = 512
    guidance_scale: Optional[float] = 8.0
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = 50
    webcam_image: Optional[str] = ""
    strength: Optional[float] = 0.8

app = FastAPI(on_startup=[on_startup])
@app.post('/api/infer')
def infer(body: InferenceArgs):
    #print(body)
    #global trt
    #trt.denoising_steps = body.num_inference_steps

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

    #change_scheduler(trt, "Scheduler.euler_a", sched_opts)
    image_bytes = base64.b64decode(body.webcam_image)
    buf = io.BytesIO(image_bytes)

    # print(type(image_bytes))
    init_image = PIL.Image.open(buf)
    init_image = PIL.ImageOps.exif_transpose(init_image)

    init_image = init_image.convert('RGB').resize((512, 512), resample=PIL.Image.LANCZOS)

    response = worker.generate(body, init_image)

    """dur, response = trt.infer(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        height=body.height,
        width=body.width,
        guidance_scale=body.guidance_scale,
        seed=body.seed,
        num_inference_steps=body.num_inference_steps,
        image=init_image,
        image_guidance_scale=body.strength,
        eta=0.0,
    )"""



    print(type(response))
    print(response)
    #img = Image.fromarray(response)
    images = image_tobytes(response)

    return StreamingResponse(io.BytesIO(packb({
        "images": images,
        "duration": 0
    })))


"""def infer(
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
    ):"""