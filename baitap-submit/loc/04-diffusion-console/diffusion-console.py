from diffusers import DiffusionPipeline
import torch
import random

seed = random.randint(0, 2**32 - 1)

pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False
)
device ="cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)

prompt = input("Input prompt: ")
width = int(input("Input width: "))
height = int(input("Input height: "))

image = pipeline(
    prompt,
    height=height,
    width=width,
    guidance_scale=6.5,
    num_inference_steps=24,
    negative_prompt="ugly, deformed, disfigured, low quality, worst quality",
    generator=torch.Generator(device=device).manual_seed(seed),
).images[0]

image.save("output.png")
image.show()