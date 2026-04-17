#!/bin/bash

# Simea server — DIMV latent reasoning, FULL fine-tuning (no LoRA), GPUs 4,5
# Sequence layout: [x_txt | V | Z | Y]
# Only NTP loss; information bottleneck enforced by the 4D attention mask.
# LLM fully trainable + LatentReasoningModule fully trainable; vision tower frozen.

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
export MASTER_PORT=29505

# Load personal wandb credentials
ENV_FILE="$REPO/.env.simea"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "[env] Loaded credentials from .env.simea"
else
    echo "[warn] .env.simea not found — wandb may use wrong account."
fi

GPU_INCLUDE="localhost:4,5"

# Checkpoint resume — set to checkpoint dir to resume, empty to start fresh
RESUME_FROM=""

# Model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-3B-DIMV-FULL-STAGE1"

# Data packing
# Conservative shared-GPU setup:
# - keep micro-batch at 1
# - halve packed-token pressure versus the original 4-GPU run
# - recover effective batch with grad accumulation instead of larger packed batches
DATA_PACKING=True
LST=4096
MAX_INSTANCE_PER_BATCH=1
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))    # 4096 tokens

RANDOM_SEED=42
DATA_PATH="/home/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_simea_filtered.json"
IMAGE_FOLDER="/home/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data"

# Training params — 2 GPUs x batch 1 x grad_accum 8 = effective batch 16
MAX_STEPS=5000
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=8

# Learning rates (full FT — lower than LoRA)
LR=2e-5
LATENT_REASONING_LR=1e-4   # higher LR for the new LatentReasoningModule

# DIMV slot config
NUM_SLOTS=64          # T_v — number of latent reasoning tokens
NUM_REFINE_STEPS=2    # L — iterative cross-attention steps
SLOT_INIT="learned"
EARLY_CHECKPOINT_STEPS="10,100"
CHECKPOINT_EVERY=500
VSTAR_VAL_FRACTION=0.30
VSTAR_VAL_SEED=42
VSTAR_CONFIGS_DIR="configs_simea"

# Image resolution
MAX_TOKEN=2048
MIN_TOKEN=128

RUN_NAME="DIMV_FULL_3B_2GPU_G45_Slots${NUM_SLOTS}_L${NUM_REFINE_STEPS}-LR${LR}-Packed${MAX_PACKED_TOKENS}-Steps${MAX_STEPS}"
OUTPUT_DIR="checkpoints_dimv/stage1_3b_dimv_full/"

START_TIME=$(date +%s)
echo "[start]  $(date) — run: $RUN_NAME"
echo "[mode]   DIMV FULL FT — no LoRA, NTP-only loss, attention-mask bottleneck"
echo "[slots]  T_v=${NUM_SLOTS} latent slots, L=${NUM_REFINE_STEPS} refinement steps, init=${SLOT_INIT}"
echo "[gpu]    GPUs 4,5 on simea (2× L40S 46GB)"
echo "[batch]  ${BATCH_PER_DEVICE} per device × ${GRAD_ACCUM_STEPS} accum × 2 GPUs = effective batch $((BATCH_PER_DEVICE * GRAD_ACCUM_STEPS * 2)) | packed_tokens=${MAX_PACKED_TOKENS}"
echo "[ckpt]   save at steps ${EARLY_CHECKPOINT_STEPS}, then every ${CHECKPOINT_EVERY}; never rotate/delete checkpoints"
echo "[eval]   fixed ${VSTAR_VAL_FRACTION} V* validation split from ${VSTAR_CONFIGS_DIR}"
echo "[resume] ${RESUME_FROM:-none (fresh start)}"

$CONDA_ENV/bin/python \
    -m deepspeed.launcher.runner --include "$GPU_INCLUDE" --master_addr 127.0.0.1 --master_port 29505 src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --coconut False \
    --dimv_mode True \
    --num_reasoning_slots $NUM_SLOTS \
    --num_refinement_steps $NUM_REFINE_STEPS \
    --slot_init $SLOT_INIT \
    --loss_lvr_lambda 0 \
    --loss_lvr_fct mse \
    --deepspeed scripts/zero3_no_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --lvr_head False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --latent_reasoning_lr $LATENT_REASONING_LR \
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
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "no" \
    --checkpoint_dir_roi "$OUTPUT_DIR" \
    --validate_every_n_steps $CHECKPOINT_EVERY \
    --early_checkpoint_steps "$EARLY_CHECKPOINT_STEPS" \
    --vstar_val_fraction $VSTAR_VAL_FRACTION \
    --vstar_val_seed $VSTAR_VAL_SEED \
    --vstar_configs_dir "$VSTAR_CONFIGS_DIR" \
    --vstar_max_new_tokens 32 \
    --dataloader_num_workers 2 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}

END_TIME=$(date +%s)
ELAPSED_S=$(( END_TIME - START_TIME ))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))
echo "[done] $(date) — elapsed: ${ELAPSED_H}h ${ELAPSED_M}m"
