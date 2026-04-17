# LVR Makefile — quick re-run commands
# Usage: make <target>   (override variables with: make eval-vstar CHECKPOINT=/path/to/ckpt)
#
# Server detection: auto-includes Makefile.maui or Makefile.simea based on hostname.
# To force a server: make <target> SERVER=simea  (but auto-detect is usually correct)

HOSTNAME := $(shell hostname)
ifneq (,$(findstring maui,$(HOSTNAME)))
  include Makefile.maui
else ifneq (,$(findstring simea,$(HOSTNAME)))
  include Makefile.simea
else
  $(warning Unknown host "$(HOSTNAME)" — defaulting to Makefile.maui. Set SERVER manually if wrong.)
  include Makefile.maui
endif

LVR_STEPS    ?= 8
OUTPUT_DIR   ?= ./eval_results
STEPS_LIST   ?= 4 8 16
QUESTION     ?= "What is the material of the glove? (A) rubber (B) cotton (C) kevlar (D) leather Answer with the option letter directly."

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

## Single-image inference using the stage1-3b checkpoint-1000
## Override: make infer-ckpt1000 IMAGE=/path/img.jpg QUESTION="..." LVR_STEPS=8
infer-ckpt1000:
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(STAGE1_3B_CKPT1000) \
		--image $(IMAGE) \
		--question $(QUESTION) \
		--lvr_steps $(LVR_STEPS) \
		--decoding_strategy steps

# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

## V* Bench evaluation (full / non-LoRA checkpoint)
eval-vstar:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark vstar \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## BLINK evaluation (auto-downloads from HuggingFace)
eval-blink:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark blink \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## MMVP evaluation
eval-mmvp:
	$(PYTHONPATH_CMD) $(PYTHON) inference.py \
		--checkpoint $(CHECKPOINT) \
		--benchmark mmvp \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## Run all three benchmarks back-to-back
eval-all: eval-vstar eval-blink eval-mmvp

## V* Bench evaluation on the last complete stage1-3b-1gpu checkpoint (step 1000)
## checkpoint-1500 was cut short by disk-full; checkpoint-1000 is the last fully-saved ckpt
STAGE1_3B_CKPT1000 ?= $(REPO_ROOT)/checkpoints_rerun/stage1_3b_1gpu/checkpoint-1000
EVAL_OUTPUT_STAGE1_3B_CKPT1000 ?= ./eval_results/stage1_3b_ckpt1000

eval-vstar-stage1-3b-ckpt1000:
	mkdir -p $(EVAL_OUTPUT_STAGE1_3B_CKPT1000)
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(STAGE1_3B_CKPT1000) \
		--benchmark vstar \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(EVAL_OUTPUT_STAGE1_3B_CKPT1000) \
		2>&1 | tee logs/eval_vstar_stage1_3b_ckpt1000.log
	@echo "[eval-vstar-stage1-3b-ckpt1000] done — results in $(EVAL_OUTPUT_STAGE1_3B_CKPT1000)"

## V* Bench evaluation on the DIMV full-FT checkpoint — sweeps L=[4,8,16,32,64]
## DIMV uses LatentReasoningModule at prefill (T_v=64, trained L=2).
## inference.py auto-detects DIMV from config and routes to run_benchmark_dimv.
## Vary DIMV_L_LIST to restrict the sweep, e.g.: make eval-vstar-stage1-3b-dimv DIMV_L_LIST="4 8"
STAGE1_3B_DIMV_CKPT ?= /home/mvu9/folder_04_ma/latent_visual/checkpoints_dimv/stage1_3b_dimv_full/checkpoint-1800
EVAL_OUTPUT_STAGE1_3B_DIMV ?= ./eval_results/stage1_3b_dimv_ckpt1800
DIMV_L_LIST ?= 2 4 8 16 32 64

eval-vstar-stage1-3b-dimv:
	mkdir -p logs $(EVAL_OUTPUT_STAGE1_3B_DIMV)
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(STAGE1_3B_DIMV_CKPT) \
		--benchmark vstar \
		--configs_dir $(CONFIGS_DIR) \
		--dimv_refinement_steps_list $(DIMV_L_LIST) \
		--output_dir $(EVAL_OUTPUT_STAGE1_3B_DIMV) \
		2>&1 | tee logs/eval_vstar_stage1_3b_dimv.log
	@echo "[eval-vstar-stage1-3b-dimv] done — results in $(EVAL_OUTPUT_STAGE1_3B_DIMV)"

## MMVP evaluation on the DIMV full-FT checkpoint — sweeps L values, reports pair accuracy
## Override: make eval-mmvp-stage1-3b-dimv DIMV_L_LIST="2 4 8" STAGE1_3B_DIMV_CKPT=/path/to/ckpt
eval-mmvp-stage1-3b-dimv:
	mkdir -p logs $(EVAL_OUTPUT_STAGE1_3B_DIMV)
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(STAGE1_3B_DIMV_CKPT) \
		--benchmark mmvp \
		--configs_dir $(CONFIGS_DIR) \
		--dimv_refinement_steps_list $(DIMV_L_LIST) \
		--output_dir $(EVAL_OUTPUT_STAGE1_3B_DIMV) \
		2>&1 | tee logs/eval_mmvp_stage1_3b_dimv.log
	@echo "[eval-mmvp-stage1-3b-dimv] done — results in $(EVAL_OUTPUT_STAGE1_3B_DIMV)"

## DIMV benchmark sweep on all three benchmarks (vstar, mmvp, blink)
eval-all-dimv:
	mkdir -p logs $(EVAL_OUTPUT_STAGE1_3B_DIMV)
	for bench in vstar mmvp blink; do \
		$(INFER_CMD) $(PYTHON) inference.py \
			--checkpoint $(STAGE1_3B_DIMV_CKPT) \
			--benchmark $$bench \
			--configs_dir $(CONFIGS_DIR) \
			--dimv_refinement_steps_list $(DIMV_L_LIST) \
			--output_dir $(EVAL_OUTPUT_STAGE1_3B_DIMV) \
			2>&1 | tee logs/eval_$${bench}_stage1_3b_dimv.log; \
	done
	@echo "[eval-all-dimv] done — results in $(EVAL_OUTPUT_STAGE1_3B_DIMV)"

## V* Bench evaluation with LoRA checkpoint (auto-detects adapter_config.json, merges weights)
## Default LORA_CHECKPOINT = checkpoints_rerun/stage1_3b_lora/checkpoint-2000 (set in Makefile.simea/maui)
eval-vstar-lora:
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(LORA_CHECKPOINT) \
		--benchmark vstar \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## BLINK evaluation with LoRA checkpoint
eval-blink-lora:
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(LORA_CHECKPOINT) \
		--benchmark blink \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## MMVP evaluation with LoRA checkpoint
eval-mmvp-lora:
	$(INFER_CMD) $(PYTHON) inference.py \
		--checkpoint $(LORA_CHECKPOINT) \
		--benchmark mmvp \
		--configs_dir $(CONFIGS_DIR) \
		--lvr_steps_list $(STEPS_LIST) \
		--output_dir $(OUTPUT_DIR)

## Run all three benchmarks with LoRA checkpoint
eval-all-lora: eval-vstar-lora eval-blink-lora eval-mmvp-lora

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Stage-1 SFT (7B)
train-stage1:
	$(PYTHONPATH_CMD) bash scripts/finetune_lvr_stage1_7b.sh

## Stage-1 SFT (3B) — uses all GPUs, logs to logs/finetune_lvr_stage1_3b.log
## Script is auto-selected per server: Makefile.maui / Makefile.simea → SCRIPT_STAGE1_3B
train-stage1-3b:
	mkdir -p logs
	$(PYTHONPATH_CMD) bash $(SCRIPT_STAGE1_3B) 2>&1 | tee logs/finetune_lvr_stage1_3b.log
	@echo "[train-stage1-3b] done (server: $(SERVER))"

## Stage-1 SFT (3B) with LoRA — ZeRO-3 offload, rank=64, logs to logs/finetune_lvr_stage1_3b_lora.log
## Script auto-selected per server via SCRIPT_STAGE1_3B_LORA
train-stage1-3b-lora:
	mkdir -p logs
	bash $(SCRIPT_STAGE1_3B_LORA) 2>&1 | tee logs/finetune_lvr_stage1_3b_lora.log
	@echo "[train-stage1-3b-lora] done (server: $(SERVER))"

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

## Stage-1 SFT (3B) on single GPU 4 — simea only, checkpoints → checkpoints_rerun/stage1_3b_1gpu/
train-stage1-3b-1gpu:
	mkdir -p logs
	bash $(SCRIPT_STAGE1_3B_1GPU) 2>&1 | tee logs/finetune_lvr_stage1_3b_1gpu.log
	@echo "[train-stage1-3b-1gpu] done (server: $(SERVER))"

## Stage-1 prototype LVR (3B + LoRA) — K=8 parallel prototype slots, GPUs 6,7, checkpoints → checkpoints_rerun/stage1_3b_prototype/
train-stage1-3b-prototype:
	mkdir -p logs
	bash $(SCRIPT_STAGE1_3B_PROTO) 2>&1 | tee logs/finetune_lvr_stage1_3b_prototype.log
	@echo "[train-stage1-3b-prototype] done (server: $(SERVER))"

## Stage-1 prototype LVR (3B, FULL fine-tuning) — K=8 prototype slots, entire LLM trainable, GPUs 4,5,6,7
train-stage1-3b-prototype-full:
	mkdir -p logs
	bash $(SCRIPT_STAGE1_3B_PROTO_FULL) 2>&1 | tee logs/finetune_lvr_stage1_3b_prototype_full.log
	@echo "[train-stage1-3b-prototype-full] done (server: $(SERVER))"

## Stage-1 DIMV latent reasoning (3B, FULL FT, no LoRA) — NTP-only loss, attention-mask bottleneck
## T_v=64 slots, L=2 refinement steps, full LLM + LatentReasoningModule trainable, GPUs 4,5,6,7
## Logs to logs/finetune_lvr_stage1_3b_dimv.log

train-stage1-3b-dimv:
	mkdir -p logs
	bash scripts/finetune_lvr_stage1_3b_dimv_full_simea.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_dimv.log
	@echo "[train-stage1-3b-dimv] done"

## Stage-1 DIMV latent reasoning (3B + LoRA) — same but with LoRA r=64 on LLM
train-stage1-3b-dimv-lora:
	mkdir -p logs
	bash scripts/finetune_lvr_stage1_3b_dimv_simea.sh 2>&1 | tee logs/finetune_lvr_stage1_3b_dimv_lora.log
	@echo "[train-stage1-3b-dimv-lora] done"

## Stage-2 RL/GRPO (7B) — edit CHKPT_PATH inside the script first
train-stage2:
	bash scripts/finetune_lvr_stage2_7b.sh

.PHONY: infer-hf infer-ckpt1000 eval-vstar eval-vstar-stage1-3b-ckpt1000 eval-vstar-stage1-3b-dimv eval-mmvp-stage1-3b-dimv eval-all-dimv eval-blink eval-mmvp eval-all eval-vstar-lora eval-blink-lora eval-mmvp-lora eval-all-lora train-stage1 train-stage1-3b train-stage1-3b-1gpu train-stage1-3b-lora train-stage1-3b-lora-zero2 train-stage1-3b-nowandb train-stage1-3b-lvrhead-nowandb train-stage1-3b-prototype train-stage1-3b-prototype-full train-stage1-3b-dimv train-stage1-3b-dimv-lora train-stage2 monitor-gpu


# # Single image
# python inference.py \
#   --checkpoint checkpoints_dimv/stage1_3b_dimv_full/checkpoint-1450 \
#   --image path/to/img.jpg \
#   --question "What is in this image?" \
#   --num_refinement_steps 8

# # Benchmark sweep over L=[4,8,16,32,64]
# make eval-vstar-stage1-3b-dimv

# # Custom L values
# make eval-vstar-stage1-3b-dimv DIMV_L_LIST="2"

# # All benchmarks
# make eval-all-dimv

# Run it with:


# make eval-mmvp-stage1-3b-dimv
# Overrides available:


# # Only L=2 (training baseline)
# make eval-mmvp-stage1-3b-dimv DIMV_L_LIST="2"

# # Custom checkpoint
# make eval-mmvp-stage1-3b-dimv STAGE1_3B_DIMV_CKPT=/path/to/ckpt

# # Custom output dir
# make eval-mmvp-stage1-3b-dimv EVAL_OUTPUT_STAGE1_3B_DIMV=./eval_results/my_run
# Results go to eval_results/stage1_3b_dimv_ckpt1450/mmvp/dimv_L{L:03d}.json and the log to logs/eval_mmvp_stage1_3b_dimv.log. The terminal output will show both per-question accuracy and the standard MMVP pair accuracy for each L value.