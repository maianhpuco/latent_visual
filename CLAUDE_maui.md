# Latent Visual Project — Claude Rules (Server: MAUI)

> **Server rule:** This project runs on two servers. Use `CLAUDE_maui.md` when on **maui**, and `CLAUDE_simea.md` when on **simea**. Each server has its own Makefile, config, and instruction file — do not mix them.
## Permissions
- You have full autonomy in this repo — no need to ask before editing, running, or installing
- Only restriction: follow the tmux session rules and never touch `*_old/` directories 

## Environment
- Python: use conda env `/project/hnguyen2/mvu9/conda_envs/latent_visual`
- **GPU: NVIDIA V100** — Flash Attention is NOT supported on V100. The model must use `sdpa` (Scaled Dot-Product Attention). Do not attempt to install or enable `flash_attn`

## Running Code — tmux Sessions
| Task | tmux Session | GPUs |
|---|---|---|
| Inference | `slidechat` | 1 GPU |
| Training | `slidechat-8gpu` | 8 GPUs |

- Use `tmux send-keys` to execute — never run code or installs directly in the shell:
```bash
  tmux send-keys -t <session> "your command here" Enter
```
- For training configs, always ensure the config is set up for **8 GPUs** (e.g. `nproc_per_node=8` or equivalent)
- All installs (`pip install`) go in `slidechat-8gpu` so packages are GPU-compatible
- If the target session is not found, tell the user instead of running locally

## File Permissions
- ✅ May create or delete files you created — always tell the user what was created and where
- ❌ `src_old/`, `scripts_old/`, `evaluation_old/` are read-only — never edit them
  - Reference them only to understand original logic when modifying `src/`, `scripts/`, `evaluation/`

## Makefile
- Append new commands to `Makefile` with a clear target name and a comment explaining what it does
- The Makefile is the canonical place to re-run any command later

## COMMAND.md
- After creating any new script or command, add usage guidance to `COMMAND.md`
- This is the single reference for how to run everything in this repo 
---

--data_path /project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/viscot_363k_lvr_formatted.json
--image_folder /project/hnguyen2/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data
 

# Data & Path Reference

All data paths are centrally defined in:
```
configs/data_path.yaml
```
Always check that file first before looking for data locations.

---

## Data Root

```
/project/hnguyen2/mvu9/datasets/lvr_data/
```

---

## Stage-1 SFT Training Data

### Formatted training JSONs (ready to use)

| File | Entries | Description |
|------|---------|-------------|
| `lvr_train/meta_data_lvr_sft_stage1.json` | 2 (index file) | **Pass this to `--data_path`**. Points to the two JSONs below. |
| `lvr_train/viscot_363k_lvr_formatted.json` | 404,120 | Main Visual-CoT SFT data, LVR-formatted |
| `lvr_train/viscot_sroie_dude_lvr_formatted.json` | — | Visual-CoT supplement (SROIE + DUDE) |

### Image path convention (Stage-1)

| Key | Path |
|-----|------|
| **JSON (pass to `--data_path`)** | `/project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/viscot_363k_lvr_formatted.json` |
| **Image folder (pass to `--image_folder`)** | `/project/hnguyen2/mvu9/datasets/lvr_data/Visual-CoT/images/cot_image_data` |

Config key: `training.visual_cot.lvr_formatted_json` and `training.visual_cot.image_folder` in `configs/data_path.yaml`.

---

## Stage-2 RL (GRPO) Training Data

### Formatted training JSON (ready to use)

| File | Entries | Description |
|------|---------|-------------|
| `lvr_train/virl39k.json` | 38,870 | **Pass this to `--data_path`** |

### Raw source (ViRL39K)
```
ViRL39K/
├── 39Krelease.parquet             # original parquet
├── images.zip                     # original archive
└── images/
    └── images/                    # ACTUAL IMAGE ROOT for stage-2
        ├── Processed-*.jpg
        └── Processed-*.png
```

### Image path convention (Stage-2)

| Key | Path |
|-----|------|
| **JSON (pass to `--data_path`)** | `/project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/virl39k.json` |
| **Image folder (pass to `--image_folder`)** | `/project/hnguyen2/mvu9/datasets/lvr_data/ViRL39K/images/images` |

Config key: `training.virl39k.lvr_formatted_json` and `training.virl39k.image_folder` in `configs/data_path.yaml`.

Note: image field in JSON is `"ViRL39K/Processed-xxx.jpg"` (no `<image>` token in Stage-2 data).

---

## Evaluation Data

### V* Bench
```
vstar_bench/
├── test_questions.jsonl           # 191 test questions
├── direct_attributes/             # image folder (JPG + JSON pairs)
│   ├── sa_10033.jpg
│   ├── sa_10033.json
│   └── ...
└── relative_position/             # image folder (JPG + JSON pairs)
```
- Config key: `evaluation.vstar_bench` in `configs/data_path.yaml`
- Questions: `vstar_bench/test_questions.jsonl`
- Image dirs: `vstar_bench/direct_attributes/` and `vstar_bench/relative_position/`

### MMVP
```
mmvp/
├── Questions.csv                  # questions + correct answers
├── Questions.xlsx
└── MMVP Images/                   # image folder (JPGs named by index)
    ├── 1.jpg
    └── ...
```
- Config key: `evaluation.mmvp` in `configs/data_path.yaml`
- Questions: `mmvp/Questions.csv`
- Image dir: `mmvp/MMVP Images/`

### BLINK (extracted)
```
blink_extracted/
├── Art_Style/
│   ├── val/
│   │   ├── records.jsonl
│   │   └── images/
│   └── test/
│       ├── records.jsonl
│       └── images/
├── Counting/        val/ + test/
├── Forensic_Detection/
├── Functional_Correspondence/
├── IQ_Test/         val/ + test/  ← records.jsonl + images/ in each
├── Jigsaw/          val/ + test/
├── Multi-view_Reasoning/
├── Object_Localization/
├── Relative_Depth/
├── Relative_Reflectance/
├── Semantic_Correspondence/
├── Spatial_Relation/
├── Visual_Correspondence/
└── Visual_Similarity/
```
- Config key: `evaluation.blink` in `configs/data_path.yaml`
- Each subtask has its own `val/records.jsonl` + `val/images/` and `test/records.jsonl` + `test/images/`
- Subtasks used in evaluation: Counting, IQ_Test, Jigsaw, Relative_Reflectance, Spatial_Relation

---

## Quick Reference: `--data_path` and `--image_folder` per stage

| Stage | `--data_path` | `--image_folder` |
|-------|--------------|-----------------|
| Stage-1 SFT | `lvr_train/meta_data_lvr_sft_stage1.json` | see warning above — needs symlink fix |
| Stage-2 RL  | `lvr_train/virl39k.json` | see warning above — needs symlink fix |

All paths are relative to `/project/hnguyen2/mvu9/datasets/lvr_data/` unless absolute.

---

## Inference

Inference script: `inference.py`  
Command reference: `COMMAND.md`  
Benchmark data paths are read automatically from `configs/data_path.yaml`.



now for all the training (new) we will save the checkpoint here: 

checkpoints_rerun/
├── stage1_7b/      ← finetune_lvr_stage1_7b.sh  (make train-stage1)
├── stage1_3b/      ← finetune_lvr_stage1_3b.sh  (make train-stage1-3b)
├── stage2_7b/      ← finetune_lvr_stage2_7b.sh  (make train-stage2)
├── stage2_3b/      ← finetune_lvr_stage2_3b.sh
├── vanilla_7b/     ← vanilla_sft_7b.sh
└── vanilla_3b/     ← vanilla_sft_3b.sh 