# LVR — Command Reference (maui server)

All commands assume you are in the repo root:
```
/project/hnguyen2/mvu9/folder_04_ma/latent_visual/
```

Server: **maui** | GPUs: V100 | flash_attn: **NOT supported** (use SDPA)

---

## 1. Environment Setup

```bash
# Python binary (use directly — no conda activate needed on maui)
/project/hnguyen2/mvu9/conda_envs/latent_visual/bin/python

# Set PYTHONPATH (add to ~/.bashrc or run before each session)
export PYTHONPATH=/project/hnguyen2/mvu9/folder_04_ma/latent_visual:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Install packages (run inside slidechat-8gpu tmux session)
```bash
tmux send-keys -t slidechat-8gpu "/project/hnguyen2/mvu9/conda_envs/latent_visual/bin/pip install <package>" Enter
```

---

## 2. tmux Sessions

| Session | Purpose | GPUs |
|---------|---------|------|
| `slidechat` | Inference | 1 GPU |
| `slidechat-8gpu` | Multi-GPU training + installs | 8 GPUs |

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
| Stage-1 3B (full fine-tune) | `checkpoints_rerun/stage1_3b/` |
| Stage-1 3B LoRA | `checkpoints_rerun/stage1_3b_lora/` |
| Stage-1 7B (full fine-tune) | `checkpoints_rerun/stage1_7b/` |
| Stage-2 7B (RL/GRPO) | `checkpoints_rerun/stage2_7b/` |

---

## 4. Training

All training targets are in the shared `Makefile` and auto-select maui scripts.

### Stage-1 SFT (3B, all GPUs)
```bash
make train-stage1-3b
# → runs scripts/finetune_lvr_stage1_3b.sh
# → logs to logs/finetune_lvr_stage1_3b.log
```

### Stage-1 SFT (3B, LoRA)
```bash
make train-stage1-3b-lora
# → runs scripts/finetune_lvr_stage1_3b_lora.sh
# → logs to logs/finetune_lvr_stage1_3b_lora.log
```

### Stage-1 SFT (7B)
```bash
make train-stage1
# → runs scripts/finetune_lvr_stage1_7b.sh
```

### Stage-2 RL / GRPO (7B)
```bash
make train-stage2
# → runs scripts/finetune_lvr_stage2_7b.sh
# Edit CHKPT_PATH inside the script first to point to a Stage-1 checkpoint
```

### Key script variables to edit
| Variable | Description |
|----------|-------------|
| `MAX_STEPS` | Default 2500 |
| `LR` | Default 1e-5 (full) or 2e-4 (LoRA) |
| `LAMBDA_LVR` | LVR loss weight (default 0.1) |
| `DATA_PATH` | Points to `meta_data_lvr_sft_stage1.json` |
| `OUTPUT_DIR` | Checkpoint output directory |

### Data paths (maui)
```
DATA_PATH    = /project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1.json
IMAGE_FOLDER = /project/hnguyen2/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data
```

---

## 5. Monitoring

### Check training logs
```bash
tail -f logs/finetune_lvr_stage1_3b_lora.log
tail -f logs/finetune_lvr_stage1_3b.log
```

### GPU utilization
```bash
make monitor-gpu
# or directly:
watch -n 2 nvidia-smi
```

---

## 6. Inference

```bash
# Via Makefile
make infer-hf CHECKPOINT=/path/to/checkpoint IMAGE=/path/to/image.jpg

# Benchmark evaluation
make eval-vstar CHECKPOINT=/path/to/checkpoint
make eval-blink CHECKPOINT=/path/to/checkpoint
make eval-mmvp  CHECKPOINT=/path/to/checkpoint
make eval-all   CHECKPOINT=/path/to/checkpoint
```

### Inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | HuggingFace repo ID or local checkpoint path |
| `--image` | — | One or more image paths |
| `--question` | — | Question string |
| `--lvr_steps` | `8` | Latent reasoning iterations (4 = fast, 8 = balanced, 16 = best) |
| `--decoding_strategy` | `steps` | `steps` = fixed iterations; `latent` = adaptive |
| `--benchmark` | — | `vstar` / `blink` / `mmvp` |
| `--output_dir` | `./eval_results` | Benchmark result output directory |

---

## 7. Notes

- **flash_attn is NOT available** on maui (V100). All scripts use `--disable_flash_attn2 True` (SDPA).
- **DeepSpeed launcher**: always use `deepspeed.launcher.runner --include localhost:...` for multi-GPU; never run `python train_lvr.py` directly.
- **8 GPU config**: ensure `nproc_per_node=8` or `--include localhost:0,1,2,3,4,5,6,7` in training scripts.
- For full data path reference, see `CLAUDE.md` and `configs/data_path.yaml`.
