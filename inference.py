from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import os


save_path='./inference_result/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model_id = "./models/1200"
g_cuda = None


g_cuda = torch.Generator(device='cuda')
seed = 1616
g_cuda.manual_seed(seed)

negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 50 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


prompt = "photo of running sks dog and hat"
save_prefix='sks_dog_hat'
with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for i in range(len(images)):
    images[i].save(os.path.join(save_path, save_prefix+f"{i}.png"))
