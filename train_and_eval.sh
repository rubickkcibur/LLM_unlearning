#!/bin/bash
#SBATCH -o /aifs4su/rubickjiang/logs/job.%j.out.log
#SBATCH --error /aifs4su/rubickjiang/logs/job.%j.err.log
#SBATCH -p batch
#SBATCH -J med-qasc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8
#SBATCH -t 7-00:00:00
#SBATCH -c 32
MODEL_PATH="/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B"
DATA_PATH="/aifs4su/rubickjiang/unlearning/data/self_generated_base/qasc/wrong_answer.jsonl"
DATASET_NAME="medmcqa"
VALID_DATA_PATH=""
OUTPUT_DIR="/aifs4su/rubickjiang/unlearning/models/medmcqa-base-de-qasc-GA-ahead-0.01-0.05"
TEMP_PATH=""
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_USE_CUDA_DSA=1
export SEED=114514

accelerate launch --config_file "/home/rubickjiang/.cache/huggingface/accelerate/default_config_ds.yaml" train_unlearning.py \
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
  --filter_model_lr 1e-3 \
  --uncertainty_th 0.8 \
  --unlearning_alpha 0.05 \
  --unlearning_portion 0.01 \
  --unlearning_loss "GA" \
  --arrange "ahead" \
  --num_train_epochs 1 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --ddp_find_unused_parameters False \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "steps" \
  --save_strategy "epoch" \
  --save_only_model True \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 4 \
  --report_to "wandb" \
  --model_max_length 1024 \
  --lazy_preprocess False \
  --use_lora False \
  --gradient_checkpointing False

dir=$(ls -d "$OUTPUT_DIR"/*/)
TEST_DATA_NAMES=("gsm8k" "medmcqa" "qasc" "svamp" "aqua")
for i in {0..4}
do
    accelerate launch --config_file "/home/rubickjiang/.cache/huggingface/accelerate/default_config_acc.yaml" evaluation.py \
    --model_name_or_path "$dir" \
    --mode "base" \
    --peft_model_path "" \
    --dataset_name "${TEST_DATA_NAMES[i]}" \
    --data_path "" \
    --valid_data_path "" \
    --eval_data_path "${TEST_DATA_NAMES[i]}:test" \
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
    --learning_rate 2e-4 \
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