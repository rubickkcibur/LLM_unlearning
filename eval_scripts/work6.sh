#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_PATH="/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B"
DATA_PATH=""
DATASET_NAME="medmcqa"
SPLIT=0
VALID_DATA_PATH=""
OUTPUT_DIR=""
TEMP_PATH=""
PEFT_MODEL=""
MODEL_PATHS[0]="/aifs4su/rubickjiang/unlearning/models/gsm8k-base-lora-de-med-1vs8-0.01/checkpoint-66"
MODEL_PATHS[1]="/aifs4su/rubickjiang/unlearning/models/gsm8k-base-lora-de-med-1vs8-0.05/checkpoint-66"
MODEL_PATHS[2]="/aifs4su/rubickjiang/unlearning/models/gsm8k-base-lora-de-med-1vs8-0.1/checkpoint-66"
MODEL_PATHS[3]="/aifs4su/rubickjiang/unlearning/models/gsm8k-base-lora-de-med-1vs8-0.15/checkpoint-66"
MODEL_PATHS[4]="/aifs4su/rubickjiang/unlearning/models/gsm8k-base-lora-de-med-1vs8-0.2/checkpoint-66"
export TORCH_USE_CUDA_DSA=1
# export CUDA_VISIBLE_DEVICES=0
# what matters: model_name_or_path, peft_model_path, eval_data_path, per_device_eval_batch_size(fixed)
export SEED=114514
for i in {0..4}
do
    accelerate launch --config_file "/home/rubickjiang/.cache/huggingface/accelerate/default_config_acc.yaml" evaluation.py \
    --model_name_or_path "$MODEL_PATH" \
    --mode "base" \
    --peft_model_path "${MODEL_PATHS[i]}" \
    --dataset_name "$DATASET_NAME" \
    --data_path "" \
    --valid_data_path "" \
    --eval_data_path "$DATASET_NAME:test" \
    --data_filter_mode "" \
    --filter_base_model_path "" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --filter_model_lr 1e-5 \
    --uncertainty_th 1.0 \
    --num_train_epochs 1 \
    --filter_training_batch_size 8 \
    --valid_batch_size 16 \
    --filter_training_epochs 30 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess False \
    --use_lora True \
    --gradient_checkpointing True
done

exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json