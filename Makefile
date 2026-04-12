# LVR Makefile — quick re-run commands
# Usage: make <target>   (override variables with: make eval-vstar CHECKPOINT=/path/to/ckpt)

PYTHON       = /project/hnguyen2/mvu9/conda_envs/latent_visual/bin/python
PYTHONPATH_CMD = PYTHONPATH=/project/hnguyen2/mvu9/folder_04_ma/latent_visual:$$PYTHONPATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default checkpoint — override on the command line as needed
# CHECKPOINT   ?= vincentleebang/LVR-7B
CHECKPOINT ?= /project/hnguyen2/mvu9/folder_04_ma/latent_visual/checkpoints/LVR-7B
LVR_STEPS    ?= 8
OUTPUT_DIR   ?= ./eval_results
STEPS_LIST   ?= 4 8 16

# For single inference — set these when calling make
IMAGE    ?= /project/hnguyen2/mvu9/datasets/lvr_data/vstar_bench/direct_attributes/sa_4690.jpg
QUESTION ?= "What is the material of the glove? (A) rubber (B) cotton (C) kevlar (D) leather Answer with the option letter directly."

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

## Run single-image inference with the HuggingFace checkpoint
infer-hf:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--image $(IMAGE) \
		--question $(QUESTION) \
		--lvr_steps $(LVR_STEPS) \
		--decoding_strategy steps

# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

## V* Bench evaluation
eval-vstar:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark vstar \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## BLINK evaluation (auto-downloads from HuggingFace)
eval-blink:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark blink \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## MMVP evaluationgit
eval-mmvp:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark mmvp \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## Run all three benchmarks back-to-back
eval-all: eval-vstar eval-blink eval-mmvp

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Stage-1 SFT (7B)
train-stage1:
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_7b.sh

## Stage-1 SFT (3B) — uses all GPUs, logs to logs/finetune_lvr_stage1_3b.log
train-stage1-3b:
	mkdir -p logs
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_3b.sh 2>&1 | tee logs/finetune_lvr_stage1_3b.log
	@echo "[train-stage1-3b] done"

## Stage-1 SFT (3B) with LoRA — ZeRO-3 offload, rank=64, logs to logs/finetune_lvr_stage1_3b_lora.log
train-stage1-3b-lora:
	mkdir -p logs
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_3b_lora.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_lora.log
	@echo "[train-stage1-3b-lora] done"

## Stage-1 SFT (3B) with LoRA + ZeRO-2 (no CPU offload — faster if it fits in VRAM)
train-stage1-3b-lora-zero2:
	mkdir -p logs
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_3b_lora_zero2.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_lora_zero2.log
	@echo "[train-stage1-3b-lora-zero2] done"

## Stage-1 SFT (3B) without wandb — logs to screen + file; GPU monitor in split pane
train-stage1-3b-nowandb:
	mkdir -p logs
	tmux split-window -v -l 20 "$(PYTHON) scripts/gpu_monitor.py; echo '[gpu_monitor] stopped — press Enter'; read" 2>/dev/null || \
		echo "[warn] could not open tmux split; run 'make monitor-gpu' in another pane"
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_3b_nowandb.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_nowandb.log
	@echo "[train-stage1-3b-nowandb] done"

## Live GPU utilisation + memory monitor (all GPUs, 2 s refresh) — run in a spare pane
monitor-gpu:
	$(PYTHON) scripts/gpu_monitor.py

## Stage-1 SFT (3B) with LVR head (simple: LayerNorm+GELU MLP), no wandb — logs to screen + file
train-stage1-3b-lvrhead-nowandb:
	mkdir -p logs
	tmux split-window -v -l 20 "$(PYTHON) scripts/gpu_monitor.py; echo '[gpu_monitor] stopped — press Enter'; read" 2>/dev/null || \
		echo "[warn] could not open tmux split; run 'make monitor-gpu' in another pane"
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_3b_lvrhead_nowandb.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_lvrhead_nowandb.log
	@echo "[train-stage1-3b-lvrhead-nowandb] done"

## Stage-2 RL/GRPO (7B) — edit CHKPT_PATH inside the script first
train-stage2:
	bash scripts/finetune_lvr_stage2_7b.sh

.PHONY: infer-hf eval-vstar eval-blink eval-mmvp eval-all train-stage1 train-stage1-3b train-stage1-3b-lora train-stage1-3b-lora-zero2 train-stage1-3b-nowandb train-stage1-3b-lvrhead-nowandb train-stage2 monitor-gpu
