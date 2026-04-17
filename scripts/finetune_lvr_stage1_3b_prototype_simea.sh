#!/bin/bash

# Simea server — Prototype-based parallel LVR training (GPUs 4,5,6,7)
# Activates K=8 parallel prototype slots instead of sequential [lvr] generation.
# z_k = g_φ(p_k, O) for all k simultaneously, ∂z_k/∂z_j = 0 (no error cascade).

REPO=/home/mvu9/folder_04_ma/latent_visual
CONDA_ENV=/home/mvu9/conda_envs/latent_visual
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=$CONDA_ENV
export PATH=$CONDA_ENV/bin:$PATH

# NCCL: force loopback interface for single-node multi-GPU
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

# Load personal wandb credentials
ENV_FILE="$REPO/.env.simea"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "[env] Loaded credentials from .env.simea"
else
    echo "[warn] .env.simea not found — wandb may use wrong account."
fi

# All 4 available GPUs: 4,5,6,7
GPU_INCLUDE="localhost:4,5,6,7"

# Model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-3B-SFT-STAGE-1-PROTOTYPE"

# Data packing
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=2
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))  # 8192 tokens

RANDOM_SEED=42
DATA_PATH="/home/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_simea.json"
IMAGE_FOLDER="/home/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data"

# Training params
MAX_STEPS=5000
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=2

# Learning rates
LR=2e-4             # LoRA adapters
PROTOTYPE_LR=5e-4   # PrototypeBank P and g_φ (higher: randomly initialised from scratch)

# Prototype architecture
NUM_PROTOTYPES=8
PROTO_NUM_HEADS=8
PROTO_DROPOUT=0.1

# Auxiliary loss weights
LOSS_DIV_LAMBDA=0.05    # L_div: prevents slot collapse (all prototypes identical)
LOSS_FOCUS_LAMBDA=0.01  # L_focus: encourages focused attention on specific patches

# Warmup: train prototype modules only for first 500 steps
WARMUP_PROTO_ONLY=500

# LVR loss (CE + lambda * proto_aux_loss)
LAMBDA_LVR=1.0   # proto_aux_loss is already λ_div * L_div + λ_focus * L_focus

# Image resolution
MAX_TOKEN=2048
MIN_TOKEN=128

# LoRA params
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

RUN_NAME="Stage1_3B_Prototype_K${NUM_PROTOTYPES}_r${LORA_RANK}_div${LOSS_DIV_LAMBDA}_focus${LOSS_FOCUS_LAMBDA}-Steps${MAX_STEPS}"
OUTPUT_DIR="checkpoints_rerun/stage1_3b_prototype/"

START_TIME=$(date +%s)
echo "[start] $(date) — run: $RUN_NAME"
echo "[gpu]   GPUs 4,5,6,7 on simea"
echo "[proto] K=${NUM_PROTOTYPES} prototypes, parallel inference, ∂z_k/∂z_j=0"
echo "[eta]   MAX_STEPS=$MAX_STEPS | ~6s/step on 4 GPUs → ETA ~$((MAX_STEPS * 6 / 3600))h"

$CONDA_ENV/bin/python \
    -m deepspeed.launcher.runner --include $GPU_INCLUDE --master_addr 127.0.0.1 --master_port 29503 src/train/train_lvr_prototype.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --prototype_mode True \
    --num_prototypes $NUM_PROTOTYPES \
    --prototype_num_heads $PROTO_NUM_HEADS \
    --prototype_dropout $PROTO_DROPOUT \
    --loss_diversity_lambda $LOSS_DIV_LAMBDA \
    --loss_focus_lambda $LOSS_FOCUS_LAMBDA \
    --warmup_steps_prototype_only $WARMUP_PROTO_ONLY \
    --loss_lvr_fct mse \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --lvr_head False \
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
    --prototype_lr $PROTOTYPE_LR \
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
    --save_total_limit 10 \
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
