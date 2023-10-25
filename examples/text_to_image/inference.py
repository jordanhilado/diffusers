import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub.repocard import RepoCard
from PIL import Image

prompts = [
        "A blue dragon pokemon flying through the air",
        "A sketch of a scene of two walking zebras in a jungle",
        "Green Arc'teryx jacket with a hood and a white logo on the front"
]

models = {
    0: {
        "name": "jordanhilado/sd-1-1-pokemon-lora",
        "prompt": prompts[0],
        "output_file": "1-1-pokemon",
    },
    1: {
        "name": "jordanhilado/sd-1-5-pokemon-lora",
        "prompt": prompts[0],
        "output_file": "1-5-pokemon",
    },
    2: {
        "name": "jordanhilado/sd-1-1-sketch-lora",
        "prompt": prompts[1],
        "output_file": "1-1-sketch",
    },
    3: {
        "name": "jordanhilado/sd-1-5-sketch-lora",
        "prompt": prompts[1],
        "output_file": "1-5-sketch",
    },
    4: {
        "name": "jordanhilado/sd-1-1-kream-lora",
        "prompt": prompts[2],
        "output_file": "1-1-kream",
    },
    5: {
        "name": "jordanhilado/sd-1-5-kream-lora",
        "prompt": prompts[2],
        "output_file": "1-5-kream",
    },
}

model_choice = 5

lora_model_id, prompt, output_file = (
    models[model_choice]["name"],
    models[model_choice]["prompt"],
    models[model_choice]["output_file"],
)


def filename(scale):
    return f"{output_file}-scale-{scale}.png"


card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, safety_checker=None)
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
