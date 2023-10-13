from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/home/dando/projects/diffusers/examples/text_to_image/sd-1-5-finetune-lora-2",
    path_in_repo="/",
    repo_id="jordanhilado/sd-1-5-pokemon-lora",
)