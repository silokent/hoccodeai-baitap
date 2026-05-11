import gradio as gr
from diffusers import DiffusionPipeline
import torch
import random

seed = random.randint(0, 2**32 - 1)

MODEL_LIST = {
    "Stable Diffusion 1.5 (Anime)": "sd-legacy/stable-diffusion-v1-5",
    "Anything V5 (Anime)": "stablediffusionapi/anything-v5",
}


pipelines = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model_name):
    if model_name not in pipelines:
        print(f"Loading model: {model_name}")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_LIST[model_name],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.to(device)
        pipelines[model_name] = pipe
    return pipelines[model_name]


def generate_image(model_name, prompt, negative_prompt, steps, cfg, seed):
    pipe = load_pipeline(model_name)

    if seed == -1:
        seed = random.randint(0, 999999)

    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        height=80,
        width=80
    ).images[0]

    return image


# UI
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Generator (Anime")

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_LIST.keys()),
                value="Anything V5 (Anime)",
                label="Chọn Model"
            )

            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, low quality, blurry"
            )

            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
            cfg = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale (CFG)")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")

            btn = gr.Button("Generate")

        with gr.Column():
            output = gr.Image(label="Result")

    btn.click(
        fn=generate_image,
        inputs=[model_dropdown, prompt, negative_prompt, steps, cfg, seed],
        outputs=output
    )

demo.launch()