import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub.repocard import RepoCard

lora_model_id = "jordanhilado/sd-1-5-pokemon-lora"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(lora_model_id)
pipe.to("cuda")

# image = pipe(
#     "A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0}
# ).images[0]

# image = pipe("A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5).images[0]
# image.save("blue_pokemon_scale_0.png")

# turn off nsfw filter
pipe.nsfw_filter = lambda x: False

prompt = "A pokemon red and white cartoon ball with an angry look on its face"

def filename(prompt, scale, iteration):
    return f"pokemon_red_white_ball_scale_{scale}_{iteration}.png"

# generate four images using only base model weights
for i in range(4):
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0}).images[0]
    image.save(filename(prompt, 0, i))

# generate four images using 0.5x lora model weights
for i in range(4):
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]
    image.save(filename(prompt, 0.5, i))

# generate four images using only the lora model weights
for i in range(4):
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 1}).images[0]
    image.save(filename(prompt, 1, i))