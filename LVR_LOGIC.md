# LVR Training Logic

This document explains the core Latent Visual Reasoning (LVR) training logic used in `train-stage1-3b-lora` (and all other stage-1 variants).

---

## 1. Image Processing

**Files:** `src/dataset/data_utils.py`, `src/train/monkey_patch_forward_lvr.py`

Images go through two stages:

### a) Dataset-side (preprocessing)
- `get_image_info()` (`data_utils.py:120-144`) loads each image and passes it through Qwen's `process_vision_info` utility
- Pixel values are constrained by `image_min_pixels` / `image_max_pixels` (e.g. 100352–1605632)
- The processor returns `pixel_values` (tensor) and `image_grid_thw` — a `[T, H, W]` grid describing how the image was tiled

### b) Model-side (forward pass)
- The vision tower encodes pixel values → visual embeddings (`monkey_patch_forward_lvr.py:238`)
- An `image_mask` identifies positions in `input_ids` that are image placeholder tokens
- Visual embeddings are scattered into the text embedding sequence at those positions (`line 265`):
  ```python
  inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
  ```

---

## 2. LVR Mechanism — Latent Visual Tokens

**Files:** `src/train/monkey_patch_forward_lvr.py`, `src/dataset/lvr_sft_dataset_packed.py`, `src/model/qwen_lvr_model.py`

### What is an LVR token?
`<|lvr|>` is a special token inserted into the sequence at positions corresponding to a **region of interest (ROI)** in the image. It acts as a "visual reasoning slot" — the model must predict visual features at this position.

### How LVR tokens are built (data pipeline)
1. Each training sample includes bounding boxes (`bboxes`) for ROIs
2. `bbox_to_token_idxs()` (`lvr_sft_dataset_packed.py:173-210`) maps each bounding box to the specific visual token indices it covers:
   - Converts pixel coordinates → 14×14 image grid (from `image_grid_thw`) → 28×28 visual token grid
3. The text conversation's `<lvr>` placeholder is expanded to:
   ```
   <|lvr_start|> <|lvr|> <|lvr|> ... <|lvr|> <|lvr_end|>
   ```
   where the number of `<|lvr|>` tokens = number of visual tokens in the ROI

### How LVR tokens are filled (forward pass)
- `lvr_mask = input_ids == self.config.lvr_id` identifies LVR positions (`monkey_patch_forward_lvr.py:277`)
- The actual visual embeddings for those ROI tokens are gathered from `image_embeds`
- They are inserted into `inputs_embeds` at the LVR positions (`line 301`):
  ```python
  inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
  ```
- This means the model sees the **real visual features** at LVR positions during training, and must learn to **reproduce** them from context

### Inference decoding strategies (`qwen_lvr_model.py:362-400`)
| Strategy | Behaviour |
|---|---|
| `steps` | Fixed N LVR iterations regardless of convergence |
| `latent` | Iterates until latent embedding converges (threshold-based) |
| `None` (vanilla) | Enters LVR mode on `<|lvr_start|>`, exits on `<|lvr_end|>` |

---

## 3. Loss Functions

**Files:** `src/trainer/lvr_trainer.py`, `src/train/monkey_patch_forward_lvr.py`

Three loss components are computed and summed:

### a) CE Loss — Language Modelling
- Standard next-token prediction cross-entropy
- LVR token positions are **masked out** (`IGNORE_INDEX`) so the model is not penalised for them (`monkey_patch_forward_lvr.py:865`):
  ```python
  shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)
  ```

### b) LVR Loss — Visual Feature Prediction
- Compares the model's hidden state at `<|lvr_start|>` positions against the **actual visual embeddings** of the ROI
- Default: **MSE loss** (configurable to MAE or cosine)
- Computed in **fp32** to avoid numerical instability (`line 875`):
  ```python
  selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)
  selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
  loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)
  ```
- Controlled by `loss_lvr_lambda` (default `0.1`)

### c) Mode Switch Loss (optional)
- Binary cross-entropy that teaches the model **when to exit** LVR mode
- Creates binary targets marking the "last LVR token before `<|lvr_end|>`"
- Only active when `mode_switch_loss=True` (off by default in current scripts)

### Final loss combination (`lvr_trainer.py:240-243`)
```
loss = loss_CE + λ_lvr × loss_LVR [+ λ_ms × loss_mode_switch]
```
With default settings: `loss = loss_CE + 0.1 × loss_MSE`

---

## 4. Model Architecture

**Files:** `src/model/qwen_lvr_model.py`, `src/model/lvr_heads.py`

### Base: QwenWithLVR
- Extends `Qwen2_5_VLForConditionalGeneration` (3B or 7B)
- Adds an optional **LVR head** — a small projection network on top of the LLM hidden states

### LVR Head variants (`lvr_heads.py`)

| Variant | Architecture |
|---|---|
| `LVRHead` (simple) | LayerNorm → Linear → GELU → Linear |
| `LVRHeadGLU` | Gated Linear Unit (gate × value, SiLU activation) — matches LLM FFN width |

The head is applied at `<|lvr_start|>` positions to project hidden states into visual embedding space (`monkey_patch_forward_lvr.py:833-839`).

When `lvr_head=False` (current scripts), the model uses its **raw hidden states** directly to predict visual features — no extra head. This is "naive LVR" mode (printed at startup: `Activated naive LVR without head mode!!!`).

### Optional: Learnable Latent End Embedding
- A learnable parameter `lvr_latent_end_emb` acts as a stopping signal during generation
- Enabled via `latent_end_token=True` in config

---

## 5. Data Packing

**Files:** `src/dataset/lvr_sft_dataset_packed.py`

### Purpose
Pack multiple short samples into one fixed-length sequence to maximise GPU utilisation and avoid padding waste.

### Meta-data index file format
The `--data_path` argument points to a JSON index file listing datasets:
```json
[
  {
    "ds_name": "viscot",
    "data_path": "/path/to/viscot_363k_lvr_formatted.json",
    "image_folder": "/path/to/images",
    "ds_type": "Q_A"
  }
]
```

### Packing algorithm (greedy bin-packing)
1. Each sample contributes `input_lengths` tokens
2. Samples are greedily assigned to buffers (bins):
   - Constraint 1: total tokens ≤ `max_packed_tokens` (8192)
   - Constraint 2: samples per bin ≤ `max_instance_per_batch` (2)
   - Long samples (> `long_seq_threshold` = 4096 tokens) get their own bin
3. A full bin is yielded as one batch item

### Collation
- Each packed item is **split back** by `input_lengths` inside the collator
- Sequences are padded to the longest in the batch
- `pixel_values` and `image_grid_thw` are concatenated across samples
- `lvr_tokens` (ROI visual token indices) are preserved as a list

---

## 6. Training Configuration (LoRA)

| Parameter | Value | Notes |
|---|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` | |
| LoRA rank | 64 | Applied to LLM attention layers |
| LoRA alpha | 128 | Effective scale = alpha/rank = 2× |
| LoRA dropout | 0.05 | |
| Frozen | vision tower, merger, LLM base | Only LoRA adapters trained |
| LR | 2e-4 | Higher than full fine-tune |
| LVR loss | MSE, λ=0.1 | |
| Max steps | 2500 | |
| DeepSpeed | ZeRO-3 offload | CPU offload for frozen params |
| GPUs | 6, 7 (simea) | 4,5 reserved for stage1-3b |

---

## Summary Diagram

```
Image → Vision Tower → visual_embeds [N_vis × D]
                              │
              bbox → token_idxs → select ROI embeds
                              │
Text sequence: [...] <|lvr_start|> <|lvr|>×K <|lvr_end|> [...]
                              │
                    replace <|lvr|> with ROI embeds
                              │
                     LLM forward pass
                              │
               ┌──────────────┴──────────────┐
          CE loss                      LVR loss (MSE)
      (text tokens)           hidden_state@lvr_start vs ROI embeds
```

The model learns to:
1. Generate correct text answers (CE loss)
2. Encode the visual ROI information in its hidden state at LVR positions (LVR loss)
