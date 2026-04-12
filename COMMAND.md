# LVR — Command Reference

All commands assume you are in the repo root:
```
/project/hnguyen2/mvu9/folder_04_ma/latent_visual/
```

---

## 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation   # must be installed last
```

Set PYTHONPATH before every session (or add to your `.bashrc`):
```bash
export PYTHONPATH=/project/hnguyen2/mvu9/folder_04_ma/latent_visual:$PYTHONPATH
```

---

## 2. Checkpoints

| Checkpoint | Location |
|------------|----------|
| Released (Stage-2 RL) | `vincentleebang/LVR-7B` on HuggingFace |
| Stage-1 local | `stage1_checkpoints/Qwen2.5-VL-7B-Instruct/<run_name>/checkpoint-2500/` |
| Stage-2 local | `stage2_checkpoints_39k/checkpoint-<N>/` |

---

## 3. Training

### Stage-1 SFT (7B)
```bash
bash scripts/finetune_lvr_stage1_7b.sh
```
Key args to edit inside the script:
- `DATA_PATH` — path to `lvr_train/meta_data_lvr_sft_stage1.json`
- `OUTPUT_DIR` — where to save checkpoints
- `MAX_STEPS` — default 2500

### Stage-2 RL / GRPO (7B)
```bash
bash scripts/finetune_lvr_stage2_7b.sh
```
Key args to edit inside the script:
- `CHKPT_PATH` — path to Stage-1 checkpoint
- `DATA_PATH` — path to `lvr_train/virl39k.json`
- `IMAGE_FOLDER` — image root for ViRL39K (see CLAUDE.md for path warning)
- `LVR_STEPS` — number of latent steps during RL (default 8)
- `TEMP` — temperature (sensitive, keep at 0.9)

---

## 4. Inference

**Script:** `inference.py`  
Works identically for Stage-1 (SFT) and Stage-2 (RL) checkpoints.

### Single image + question

```bash
# Using released HuggingFace checkpoint (Stage-2 RL)
python inference.py \
  --checkpoint vincentleebang/LVR-7B \
  --image /path/to/image.jpg \
  --question "What is the relative depth of the two objects?" \
  --lvr_steps 8 \
  --decoding_strategy steps

# Using a local Stage-1 SFT checkpoint
python inference.py \
  --checkpoint stage1_checkpoints/Qwen2.5-VL-7B-Instruct/.../checkpoint-2500 \
  --image /path/to/image.jpg \
  --question "Your question here" \
  --lvr_steps 8

# Multiple images (BLINK-style)
python inference.py \
  --checkpoint vincentleebang/LVR-7B \
  --image img1.jpg img2.jpg \
  --question "Which image shows more depth contrast? A. First  B. Second" \
  --lvr_steps 8
```

### Benchmark evaluation

Results are cached as JSON — delete the output file to force re-evaluation.

```bash
# V* Bench (local data via configs/data_path.yaml)
python inference.py \
  --checkpoint vincentleebang/LVR-7B \
  --benchmark vstar \
  --lvr_steps_list 4 8 16 \
  --output_dir ./eval_results

# BLINK (auto-downloaded from HuggingFace: Counting, IQ_Test, Jigsaw, Relative_Reflectance, Spatial_Relation)
python inference.py \
  --checkpoint vincentleebang/LVR-7B \
  --benchmark blink \
  --lvr_steps_list 4 8 16 \
  --output_dir ./eval_results

# MMVP (local data via configs/data_path.yaml)
python inference.py \
  --checkpoint vincentleebang/LVR-7B \
  --benchmark mmvp \
  --lvr_steps_list 4 8 16 \
  --output_dir ./eval_results
```

### Inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | HuggingFace repo ID or local checkpoint path |
| `--image` | — | One or more image paths / URLs (single inference only) |
| `--question` | — | Question string (single inference only) |
| `--lvr_steps` | `8` | Latent reasoning iterations. Try 4 (fast) / 8 (balanced) / 16 (strongest) |
| `--decoding_strategy` | `steps` | `steps` = fixed iterations; `latent` = adaptive |
| `--lvr_steps_list` | `4 8 16` | Benchmark only — evaluates at each step count |
| `--max_new_tokens` | `512` | Max tokens to generate after latent reasoning |
| `--benchmark` | — | `vstar` / `blink` / `mmvp` — run full benchmark |
| `--output_dir` | `./eval_results` | Where to save benchmark result JSONs |

### Output format

Single inference prints:
```
Model output:
<|lvr_start|><|lvr|>...<|lvr_end|><answer> A </answer>
Extracted answer: A
```

Benchmark result files: `eval_results/<benchmark>/<strategy><steps>.json`
```json
{
  "question_id": 42,
  "prediction": "<|lvr_start|>...<answer>A</answer>",
  "label": "A",
  "category": "direct_attributes"
}
```

---

## 5. Makefile shortcuts

```bash
make infer-hf        # single inference with released HF checkpoint (edit IMAGE/QUESTION vars)
make eval-vstar      # V* benchmark with HF checkpoint
make eval-blink      # BLINK benchmark with HF checkpoint
make eval-mmvp       # MMVP benchmark with HF checkpoint
```

See `Makefile` for the full targets and how to override variables.

---

## Notes

- The same `inference.py` works for both SFT and RL checkpoints — it reads `config.lvr_head`
  from the checkpoint to auto-configure itself.
- LVR is sensitive to `--lvr_steps`. Start with 8, then compare 4 and 16.
- If you get import errors, make sure `PYTHONPATH` is set (see Section 1).
- Data path issues for training: see the warnings in `CLAUDE.md`.
