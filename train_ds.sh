#!/bin/bash
MODEL_PATH="/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B-Instruct"
DATA_PATH=""
DATASET_NAME="gsm8k"
VALID_DATA_PATH=""
OUTPUT_DIR="/aifs4su/rubickjiang/unlearning/models/gsm8k"
TEMP_PATH=""
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

accelerate launch train_ds.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "" \
  --dataset_name "$DATASET_NAME" \
  --temp_data_path "$TEMP_PATH" \
  --valid_data_path "$VALID_DATA_PATH" \
  --eval_data_path "" \
  --data_filter_mode "Self" \
  --filter_base_model_path "" \
  --bf16 True \
  --remove_unused_columns False \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 4 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_only_model True \
  --learning_rate 1e-8 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 1024 \
  --lazy_preprocess False \
  --use_lora False \
  --gradient_checkpointing False

exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json