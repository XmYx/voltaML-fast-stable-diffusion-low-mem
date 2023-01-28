import PIL
import requests
import time
from infer import DemoDiffusion

url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

trt_model = DemoDiffusion(
                model_path="engine/pix2pix"
            )

image = download_image(url)
prompt = ["make the mountains snowy"]

image_guidance_scale=1.5
guidance_scale=7
height=512
width=512
num_inference_steps=20
eta=0.0
seed=42
negative_prompt = None
num_images_per_prompt=1
# scheduler = "EulerAncestralDiscreteScheduler"

for _ in range(10):
    images = trt_model.infer(
                    prompt=prompt,
                    image=image,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                    eta=eta,
                    seed=seed
                    # scheduler
                )

start = time.time()
images = trt_model.infer(
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                eta=eta,
                seed=seed
                # scheduler
            )
print(time.time()-start)
