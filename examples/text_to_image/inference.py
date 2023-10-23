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

lora_model_id = models[0]

prompt = "A pokemon red and white cartoon ball with an angry look on its face"


def filename(prompt, scale):
    return f"pokemon_red_white_ball_scale_{scale}.png"


########################################


card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
)
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

# generate four images using only base model weights
scale_0_images = []
for i in range(4):
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0}).images[0]
    scale_0_images.append(image)
    # image.save("generated_images/" + filename(prompt, 0, i))

# crop all four images in scale_0_images into a single image using Pillow
width, height = scale_0_images[0].size

# Create a new image with the dimensions for the collage
collage_width = 2 * width  # Combine two images in each row
collage_height = 2 * height  # Combine two images in each column
collage = Image.new("RGB", (collage_width, collage_height))

# Paste the four images onto the collage
collage.paste(scale_0_images[0], (0, 0))
collage.paste(scale_0_images[1], (width, 0))
collage.paste(scale_0_images[2], (0, height))
collage.paste(scale_0_images[3], (width, height))

# Save the collage as a new image
collage.save("generated_images/" + filename(prompt, 0))

# # generate four images using 0.5x lora model weights
# for i in range(4):
#     image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]
#     image.save("generated_images/" + filename(prompt, 0.5, i))

# # generate four images using only the lora model weights
# for i in range(4):
#     image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 1}).images[0]
#     image.save("generated_images/" + filename(prompt, 1, i))
