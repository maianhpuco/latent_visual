#!/bin/bash

# Load CUDA toolkit so deepspeed can find nvcc
module load cudatoolkit/12.4 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/share/apps/cudatoolkit-12.4}

# Ensure src/ and src/train/ are on PYTHONPATH so bare imports like
# "from train.train_utils" and "from monkey_patch_forward_lvr" resolve correctly
REPO=/project/hnguyen2/mvu9/folder_04_ma/latent_visual
export PYTHONPATH=$REPO:$REPO/src:$REPO/src/train:$PYTHONPATH

# model configs
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Disable wandb entirely — results printed to stdout
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Data Config
DATA_PACKING=True

LST=4096
MAX_INSTANCE_PER_BATCH=4
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))

RANDOM_SEED=42
DATA_PATH="/project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_fixed.json"
IMAGE_FOLDER="/project/hnguyen2/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data"

# General training params
MAX_STEPS=2500
BATCH_PER_DEVICE=1            # if use data packing, BS should always be 1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=8

# LLM-related params
LR=1e-5

# LVR head — enabled, using LVRHead (simple: LayerNorm + 2-layer GELU MLP)
# Options: "simple" (LVRHead) | "glu" (LVRHeadGLU)
LVR_HEAD=True
LVR_HEAD_TYPE=simple
LVR_HEAD_LR=1e-4              # separate LR for the LVR head (higher than backbone LR)

# LVR-related params
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=2560 #5120
MIN_TOKEN=128

RUN_NAME="Stage1_3B_LVRHead${LVR_HEAD_TYPE}_${LVR_LOSS_FCT}Lambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
ONLINE=False
OUTPUT_DIR="checkpoints_rerun/stage1_3b_lvrhead/"

# if continue training, set checkpoint_name = checkpoint to continue;
# --checkpoint_name checkpoint-1400

/project/hnguyen2/mvu9/conda_envs/latent_visual/bin/python \
    -m deepspeed.launcher.runner --num_gpus $NUM_DEVICES src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --lvr_head_type $LVR_HEAD_TYPE \
    --lvr_head_lr $LVR_HEAD_LR \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --online_checkpoint $ONLINE \
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
    --report_to none \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --dataloader_num_workers 2 \
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
    # save_total_limit is for local storage only, no limit for online checkpointing
