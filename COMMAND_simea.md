# LVR — Command Reference (simea server)

All commands assume you are in the repo root:
```
/home/mvu9/folder_04_ma/latent_visual/
```

Server: **simea** | GPUs available for LVR: **4, 5, 6, 7** (0–3 are reserved, do not use)

---

## 1. Environment Setup

```bash
# Activate conda env
conda activate /home/mvu9/conda_envs/latent_visual
# or set PATH directly:
export PATH=/home/mvu9/conda_envs/latent_visual/bin:$PATH

# Set PYTHONPATH (add to ~/.bashrc or run before each session)
export PYTHONPATH=/home/mvu9/folder_04_ma/latent_visual:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/home/mvu9/conda_envs/latent_visual
```

### WandB credentials
Personal credentials are stored in `.env.simea` (gitignored).  
All training scripts auto-load them — no manual `wandb login` needed.
```
WANDB_API_KEY=<your_key>
WANDB_ENTITY=maianhpuco
```

### Install packages (run inside slidechat-8gpu tmux session)
```bash
tmux send-keys -t slidechat-8gpu "/home/mvu9/conda_envs/latent_visual/bin/pip install <package>" Enter
```

---

## 2. tmux Sessions

| Session | Purpose | GPUs |
|---------|---------|------|
| `slidechat` | Inference / 1-GPU jobs | GPU 4 |
| `slidechat-8gpu` | Multi-GPU training + installs | GPUs 4,5,6,7 |

```bash
# Attach to a session
tmux attach -t slidechat
tmux attach -t slidechat-8gpu

# Send a command without attaching
tmux send-keys -t slidechat-8gpu "your command here" Enter
```

---

## 3. Checkpoints

New checkpoints save to `checkpoints_rerun/`:

| Run | Output dir |
|-----|------------|
| Stage-1 3B (2-GPU, full fine-tune) | `checkpoints_rerun/stage1_3b/` |
| Stage-1 3B LoRA (GPUs 6,7) | `checkpoints_rerun/stage1_3b_lora/` |
| Stage-1 3B 1-GPU (GPU 4) | `checkpoints_rerun/stage1_3b_1gpu/` |

---

## 4. Training

All training targets are in the shared `Makefile` and auto-select simea scripts.

### Stage-1 SFT (3B, 2 GPUs — GPUs 4 and 5)
```bash
make train-stage1-3b
# → runs scripts/finetune_lvr_stage1_3b_simea.sh
# → logs to logs/finetune_lvr_stage1_3b.log
# → ~18-20 h for 2500 steps at ~9s/step (2 GPUs)
```

### Stage-1 SFT (3B, LoRA — GPUs 6 and 7)
```bash
make train-stage1-3b-lora
# → runs scripts/finetune_lvr_stage1_3b_lora_simea.sh
# → logs to logs/finetune_lvr_stage1_3b_lora.log
# → ~7h for 2500 steps at ~9.5s/step (2 GPUs, LoRA rank=64)
```

### Stage-1 SFT (3B, single GPU — GPU 4 only)
```bash
make train-stage1-3b-1gpu
# → runs scripts/finetune_lvr_stage1_3b_simea_1gpu.sh
# → logs to logs/finetune_lvr_stage1_3b_1gpu.log
# → ~14h for 2500 steps at ~21s/step (1 GPU, grad_accum=8)
```

### Key script variables to edit
| Variable | Location | Description |
|----------|----------|-------------|
| `MAX_STEPS` | all scripts | Default 2500 |
| `LR` | all scripts | Default 1e-5 (full) or 2e-4 (LoRA) |
| `LAMBDA_LVR` | all scripts | LVR loss weight (default 0.1) |
| `DATA_PATH` | all scripts | Points to `meta_data_lvr_sft_stage1_simea.json` |
| `OUTPUT_DIR` | all scripts | Checkpoint output directory |

### Data paths (simea)
```
DATA_PATH  = /home/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_simea.json
IMAGE_FOLDER = /home/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data
```

---

## 5. Monitoring

### Check training logs
```bash
tail -f logs/finetune_lvr_stage1_3b_lora.log
tail -f logs/finetune_lvr_stage1_3b.log
tail -f logs/finetune_lvr_stage1_3b_1gpu.log
```

### GPU utilization
```bash
make monitor-gpu
# or directly:
watch -n 2 nvidia-smi
```

### WandB dashboards
- LoRA run: https://wandb.ai/maianhpuco/LVR-Qwen25-VL-3B-SFT-STAGE-1-LORA
- 1-GPU / 3B runs: https://wandb.ai/maianhpuco/LVR-Qwen25-VL-3B-SFT-STAGE-1-450k

---

## 6. Inference

```bash
# Single image inference (GPU 4)
tmux send-keys -t slidechat "cd /home/mvu9/folder_04_ma/latent_visual && \
  PYTHONPATH=/home/mvu9/folder_04_ma/latent_visual:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=4 \
  /home/mvu9/conda_envs/latent_visual/bin/python inference.py \
  --checkpoint checkpoints_rerun/stage1_3b_lora/checkpoint-2500 \
  --image /path/to/image.jpg \
  --question 'Your question here' \
  --lvr_steps 8 \
  --decoding_strategy steps" Enter

# Or via Makefile (uses CHECKPOINT and IMAGE variables):
make infer-hf CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-2500 IMAGE=/path/to/image.jpg
```

### Benchmarks
```bash
make eval-vstar CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-2500
make eval-blink CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-2500
make eval-mmvp  CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-2500
make eval-all   CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-2500
```

---

## 7. Notes

- **flash_attn** is available on simea but currently disabled (`--disable_flash_attn2 True`) due to ABI mismatch with torch 2.6.0. All scripts use SDPA instead.
- **NCCL**: scripts set `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1` to avoid socket errors on simea's network interface.
- **DeepSpeed launcher**: always use `deepspeed.launcher.runner --include localhost:X,Y` — never run `python train_lvr.py` directly (causes MPI/ZeRO-3 errors).
- **GPU conflicts**: stage1-3b uses GPUs 4,5; lora uses GPUs 6,7; 1-GPU uses GPU 4. Do not run stage1-3b and 1-GPU at the same time.
