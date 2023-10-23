import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub.repocard import RepoCard
from PIL import Image

models = [
    "jordanhilado/sd-1-1-pokemon-lora",
    "jordanhilado/sd-1-5-pokemon-lora",
    "jordanhilado/sd-1-1-sketch-scene",
    "jordanhilado/sd-1-1-kream-lora",
]

lora_model_id = models[1]

prompt = "A pokemon red and white cartoon ball with an angry look on its face"


def filename(scale):
    return f"pokemon_red_white_ball_scale_{scale}.png"


card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(lora_model_id)
pipe.to("cuda")

pipe.nsfw_filter = lambda x: False

crop = {
    0: (0, 0),
    1: (512, 0),
    2: (0, 512),
    3: (512, 512),
}

scales = [0, 0.5, 1]

for scale in scales:
    scale_collage = Image.new("RGB", (1024, 1024))
    for i in range(4):
        image = pipe(
            prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": scale}
        ).images[0]
        scale_collage.paste(image, crop[i])
    scale_collage.save("generated_images/" + filename(scale))
