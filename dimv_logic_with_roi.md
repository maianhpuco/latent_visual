# DIMV-ROI Training Logic

## Overview

This run fine-tunes **Qwen2.5-VL-3B-Instruct** using a DIMV-inspired (Deep Imputation for Missing Views) latent reasoning approach with ROI-based supervision. The core idea is to train a model that reasons visually through a compressed bottleneck of **latent tokens Z**, rather than attending directly to raw image patches during text generation.

The total loss is:

```
L = L_NTP + λ * L_IMP
```

where `L_NTP` is the standard next-token prediction loss and `L_IMP` is an imputation loss that forces the latent tokens to encode region-of-interest visual information.

---

## Sequence Layout

```
[ x_txt | V | Z | Y ]
```

| Segment | Description |
|---------|-------------|
| `x_txt` | Text instruction/question tokens |
| `V`     | Visual patch tokens from the vision tower (~784 tokens for typical resolution) |
| `Z`     | T_v=64 latent reasoning slots (the bottleneck) |
| `Y`     | Answer tokens (NTP target) |

A **4D attention bottleneck mask** ensures that `Y` cannot attend directly to `V`. Gradients for the answer tokens flow exclusively through `Z`, forcing the model to compress visual understanding into the latent slots.

---

## DIMV Analogy

Classical DIMV (linear Gaussian) imputes missing views as:

```
E[X_miss | X_obs] = Σ_mo (Σ_oo + αI)^{-1} X_obs
```

Here, the learned nonlinear analog is:

```
Z^(final) = Attn(Q=Z, K=X_o, V=X_o)
```

where `X_o = concat(V, x_txt)` is the observed context. The cross-attention weight matrix `A ∈ R^{T_v × N_obs}` plays the role of `Σ_mo (Σ_oo + αI)^{-1}`: it learns which context tokens matter for each slot.

---

## LatentReasoningModule: Iterative Refinement

The module runs `L=2` refinement steps. At each step:

```
Q = LayerNorm(Z^(t))
Z_attended, attn_weights = CrossAttn(Q=Q, K=X_o, V=X_o)
Z_update = FFN(Z_attended)          # 4x expansion, GELU
Z^(t+1) = LayerNorm(Z^(t) + Z_update)
```

Key properties:
- All T_v=64 slots update **in parallel** — no slot attends to another slot in this module
- Slot interactions happen later in the LLM's own self-attention over the full sequence
- Slots are initialised from a **learned parameter** `Z^(0) ∈ R^{T_v × d}` (shared, trained)
- L=2 allows one correction after the initial read from context

---

## ROI Supervision (L_IMP)

Standard DIMV uses NTP loss alone. The ROI extension adds a second signal: the final latent slots `Z^(final)` must match the visual embeddings of the **region of interest** (bounding box).

```
L_IMP = d(Z^(final), pool(V_ROI*))
```

### ROIPooler

The pooler compresses `K` ROI visual tokens `V_ROI* ∈ R^{K×d}` down to `T_v` targets `Ẑ_ROI ∈ R^{T_v×d}`. This run uses `attention_pool`:

```python
Q_pool = learned_queries  # [T_v, d] — trained parameters
Ẑ_ROI, _ = MultiheadAttention(Q=Q_pool, K=V_ROI, V=V_ROI)
```

The pooler is **only used during training** to construct the imputation target. At inference time it is not needed.

### Imputation Loss

This run uses `cosine` distance with `λ=0.1`:

```
L_IMP = mean(1 - cosine_similarity(Z^(final), Ẑ_ROI))
```

Other supported options: `mse`, `nce` (noise-contrastive, with temperature `τ=0.07`).

---

## What Is Frozen vs. Trained

| Component | Status |
|-----------|--------|
| Vision tower (ViT) | **Frozen** |
| Vision-language merger | **Frozen** |
| LLM (Qwen2.5 3B transformer) | **Trainable** (full FT, no LoRA) |
| LatentReasoningModule | **Trainable** |
| ROIPooler | **Trainable** |

Two learning rates are used:
- LLM: `lr=2e-5`
- LatentReasoningModule + ROIPooler: `lr=1e-4` (higher, as these modules train from scratch)

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen2.5-VL-3B-Instruct | |
| GPUs | 2× (indices 6,7) on simea | L40S 46GB |
| Precision | BF16 | Flash-attn disabled, uses SDPA |
| Optimizer | AdamW, weight_decay=0.1 | |
| LR schedule | Cosine with 3% warmup | |
| Effective batch | 1 per device × 4 grad_accum × 2 GPUs = **8** | |
| Max steps | 5000 | |
| Data packing | Enabled, 4096 tokens/pack | 1 instance per pack |
| DeepSpeed | ZeRO-3, no CPU offload | |
| Gradient checkpointing | Enabled | |

---

## Data Pipeline

Training data: `meta_data_lvr_sft_stage1_simea_filtered.json` with Visual-CoT images.

The dataset uses `make_packed_supervised_data_module_dimv_roi`, which reads bounding boxes from the metadata (`lvr_tokens`) and constructs:
- The packed sequence `[x_txt | V | Z | Y]`
- ROI visual tokens `V_ROI*` (from bbox crops) as supervision targets for `L_IMP`

Packing bundles samples up to `MAX_PACKED_TOKENS=4096` to avoid padding waste.

---

## Special Tokens

Four new tokens are added to the tokenizer:

| Token | Role |
|-------|------|
| `<\|lvr_start\|>` | Marks start of latent reasoning block |
| `<\|lvr\|>` | Individual latent slot token (×T_v) |
| `<\|lvr_latent_end\|>` | Marks end of Z block (triggers bottleneck mask) |
| `<\|lvr_end\|>` | Marks end of latent reasoning region |

---

## Validation

Every 500 steps (and at early steps 10, 50, 100), the trainer:
1. Saves a checkpoint to `checkpoints_dimv_roi/stage1_3b_dimv_roi_full/`
2. Runs V* visual QA benchmark on a fixed 30% validation split (seeded, created once at startup)
3. Reports metrics to WandB project `LVR-Qwen25-VL-3B-DIMV-ROI-STAGE1`

Checkpoints are never deleted or rotated.

---

## Monkey-patches Applied

The script applies two monkey-patches at model load time:

1. **`replace_qwen2_5_with_mixed_modality_forward_lvr`** — replaces Qwen's standard forward pass with a mixed-modality version that inserts the `Z` slots into the sequence and applies the bottleneck attention mask.

2. **`replace_qwen_2_5_vl_patch_emb`** — patches the patch embedding layer to handle dynamic image resolutions.

3. **`replace_train_dataloader`** — replaces HuggingFace's default dataloader with a custom one that handles the packed DIMV-ROI collation format.
