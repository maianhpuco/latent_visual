#!/bin/bash

# Simea server version — flash_attn enabled, simea paths, GPUs 4,5,6,7

REPO=/home/mvu9/folder_04_ma/latent_visual
CONDA_ENV=/home/mvu9/conda_envs/latent_visual
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=$CONDA_ENV
export PATH=$CONDA_ENV/bin:$PATH

# NCCL: force loopback interface for single-node multi-GPU (avoids wrong network interface)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# Load personal wandb credentials (overrides ~/.netrc)
ENV_FILE="$REPO/.env.simea"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "[env] Loaded credentials from .env.simea"
else
    echo "[warn] .env.simea not found — wandb may use wrong account."
fi

# GPUs 0-3 are reserved on simea; GPUs 4,5 used by train-stage1-3b → use 6,7 for LoRA
GPU_INCLUDE="localhost:6,7"

# Model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-3B-SFT-STAGE-1-LORA"

# Data packing
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=2
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))   # 8192 tokens

RANDOM_SEED=42
DATA_PATH="/home/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_simea.json"
IMAGE_FOLDER="/home/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data"

# Training params
MAX_STEPS=5000
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=4

# LR
LR=2e-4
LVR_HEAD=False

# LVR loss
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

# Image resolution
MAX_TOKEN=2048
MIN_TOKEN=128

# LoRA params
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

RUN_NAME="Stage1_3B_LoRA_r${LORA_RANK}_${LVR_LOSS_FCT}Lambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-Steps${MAX_STEPS}"
OUTPUT_DIR="checkpoints_rerun/stage1_3b_lora/"

# Print ETA helper — logs start time so we can estimate finish
START_TIME=$(date +%s)
echo "[start] $(date) — run: $RUN_NAME"
echo "[eta] MAX_STEPS=$MAX_STEPS (resumes from checkpoint-2000, ~3000 steps remaining) | ~9.5s/step → ETA ~$(( (MAX_STEPS - 2000) * 10 / 3600 ))h"

$CONDA_ENV/bin/python \
    -m deepspeed.launcher.runner --include $GPU_INCLUDE --master_addr 127.0.0.1 --master_port 29501 src/train/train_lvr_lora.py \
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

END_TIME=$(date +%s)
ELAPSED_S=$(( END_TIME - START_TIME ))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))
echo "[done] $(date) — elapsed: ${ELAPSED_H}h ${ELAPSED_M}m"
