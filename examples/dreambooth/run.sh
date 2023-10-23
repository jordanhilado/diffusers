export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/jhilado/projects/dogs"
export OUTPUT_DIR="jordanhilado/sd-1-5-pokemon-lora"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
