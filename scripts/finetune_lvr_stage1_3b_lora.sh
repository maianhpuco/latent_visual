#!/bin/bash

# Load CUDA toolkit so deepspeed can find nvcc
module load cudatoolkit/12.4 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/share/apps/cudatoolkit-12.4}

REPO=/project/hnguyen2/mvu9/folder_04_ma/latent_visual
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-3B-SFT-STAGE-1-LORA"

# Data packing
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=2
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))   # 8192 tokens — ZeRO-3 offload frees frozen base model from GPU

RANDOM_SEED=42
DATA_PATH="/project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_fixed.json"
IMAGE_FOLDER="/project/hnguyen2/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data"

# Training params
MAX_STEPS=2500
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=4

# LR
LR=2e-4          # LoRA typically uses higher LR than full fine-tuning
LVR_HEAD=False

# LVR loss
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

# Image resolution
MAX_TOKEN=2048
MIN_TOKEN=128

# LoRA params
LORA_RANK=64
LORA_ALPHA=128   # alpha = 2 * rank is a common default
LORA_DROPOUT=0.05

RUN_NAME="Stage1_3B_LoRA_r${LORA_RANK}_${LVR_LOSS_FCT}Lambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}"
OUTPUT_DIR="checkpoints_rerun/stage1_3b_lora/"

/project/hnguyen2/mvu9/conda_envs/latent_visual/bin/python \
    -m deepspeed.launcher.runner --num_gpus $NUM_DEVICES src/train/train_lvr_lora.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm True \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_bias none \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --online_checkpoint False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH
