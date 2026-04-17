# DIMV-ROI: ROI-Supervised Latent Reasoning Tokens

## Overview

This document modifies the existing **DIMV-like baseline** (latent reasoning tokens
with attention mask bottleneck) to add **explicit ROI supervision**.

### The Core Shift

| | Current DIMV Baseline | DIMV-ROI (this change) |
|---|---|---|
| Z supervision | None — only NTP gradient through mask | **L_IMP: Z^(final) must match V_ROI*** |
| Z purpose | Compress image for answer generation | **Impute the ROI visual content + answer** |
| Ground truth for Z | Implicit (NTP loss only) | **Explicit: ROI visual embeddings V_ROI*** |
| DIMV analogy | Missing vars inferred from observed | **Missing vars = ROI; observed = full image + text** |

### Why This Makes Sense

In DIMV:
- **Observed variables** X_o = full image patches V + text prefix
- **Missing variables** Z = the ROI region embeddings that must be inferred

The ROI bounding box defines exactly which visual tokens are "missing" from
the context (the model sees the full image but must specifically encode the
ROI content into Z). The imputation loss makes this explicit:

```
Z^(final) ≈ V_ROI*  (imputed ≈ ground truth ROI embeddings)
```

This gives Z both:
1. Explicit visual grounding (imputation loss pulls Z toward ROI content)
2. Answer relevance (NTP loss pulls Z toward task-useful information)

### Loss Structure

```
L = L_NTP + λ * L_IMP

L_NTP = -Σ log p(y_t | Z, Q)          answer quality (through mask)
L_IMP = d(Z^(final), pool(V_ROI*))    imputation quality (Z ≈ ROI)
```

Where `d` is cosine distance or MSE, and `pool` reduces K ROI tokens to T_v slots.

---

## CRITICAL: Checkpoint Rules

```
⚠️  DO NOT DELETE ANY EXISTING CHECKPOINTS
⚠️  DO NOT OVERWRITE ANY EXISTING CHECKPOINTS
⚠️  ALL NEW CHECKPOINTS SAVED TO: checkpoints_dimv_roi/
⚠️  EXISTING checkpoints/ FOLDER: READ ONLY — NEVER MODIFY
```

After saving each checkpoint:
1. Load the fixed validation set from `checkpoints_dimv_roi/vstar_val_indices.json`
2. Run inference on the fixed 30% v*star sample
3. Save validation results to `checkpoints_dimv_roi/val_results/step_{N}.json`
4. Log validation metrics to wandb/tensorboard

---

## Notation Reference

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `I` | H×W×3 | Input image |
| `Q` | string | Natural language question |
| `Y` | sequence | Ground-truth answer tokens |
| `V` | [N, d] | All visual patch embeddings. N=784, d=2048 (Qwen 3B) |
| `x_txt` | [N_q, d] | Text prefix token embeddings |
| `X_o` | [N+N_q, d] | Observed context = concat(V, x_txt) |
| `bbox` | (x1,y1,x2,y2) | Annotated bounding box in pixel coords |
| `I_ROI` | list[int] | Flat indices of ROI patches in the 28×28 grid |
| `K` | scalar | Number of ROI visual tokens = `len(I_ROI)`. Varies per sample. |
| `V_ROI*` | [K, d] | Ground-truth ROI visual embeddings = V[I_ROI]. These are the "missing variables" Z must learn to impute. |
| `T_v` | scalar | Number of reasoning slots (fixed). Default=64. T_v << N. |
| `Z^(0)` | [T_v, d] | Initial slot embeddings (learned queries) |
| `Z^(t)` | [T_v, d] | Slot embeddings at refinement step t |
| `Z^(final)` | [T_v, d] | Final imputed reasoning tokens after L steps |
| `Ẑ_ROI` | [T_v, d] | Pooled/projected V_ROI* → T_v vectors. The imputation target. |
| `L` | scalar | Number of refinement iterations. Default=2 |
| `λ` | scalar | Weight for imputation loss. Default=0.1 |
| `d(·,·)` | scalar | Distance metric: cosine distance or MSE |

---

## Files to Modify

```
src/
├── model/
│   └── latent_reasoning_module.py     ← ADD ROIPooler, update forward signature
├── train/
│   └── monkey_patch_forward_lvr.py    ← ADD ROI extraction + imputation loss signal
├── trainer/
│   └── lvr_trainer.py                 ← ADD L_IMP computation + checkpoint logic
├── dataset/
│   └── lvr_sft_dataset_packed.py      ← ADD roi_token_idxs to collated batch
├── config/
│   └── latent_reasoning_config.py     ← ADD imputation loss fields
└── eval/
    └── vstar_validator.py             ← CREATE: fixed-set v*star evaluation
```

---

## Step 1 — Update Config

**File:** `src/config/latent_reasoning_config.py`
**Action:** Add imputation loss fields to the existing `LatentReasoningConfig`.
**Do NOT remove any existing fields.**

```python
# ADD these fields to the existing LatentReasoningConfig dataclass.
# Insert after the existing fields. Do not remove or rename existing ones.

# ── ROI imputation supervision ─────────────────────────────────────────
use_roi_supervision: bool = True
"""
If True: add L_IMP loss — Z^(final) must match pooled ROI visual embeddings.
If False: fall back to NTP-only (original DIMV baseline behaviour).
Set to False for ablation to compare with and without ROI supervision.
"""

roi_pool_method: str = "attention_pool"
"""
How to reduce K ROI visual tokens to T_v imputation targets.

K varies per sample (size of bounding box in token grid).
T_v is fixed (number of reasoning slots).
We must map K → T_v to compute the per-slot imputation loss.

Options:
- "attention_pool":  A small learned cross-attention that maps K → T_v.
                     Queries are the T_v slot positions, keys/values are V_ROI*.
                     Most expressive — each target slot can attend to all K ROI tokens.
                     Recommended default.
- "avg_pool":        Uniform average of all K ROI tokens → repeat T_v times.
                     All slots share the same scalar target.
                     Simplest, but loses spatial structure within the ROI.
- "adaptive_pool":   PyTorch adaptive average pooling K → T_v.
                     Preserves spatial ordering within the ROI.
                     Works well when T_v < K. Fails gracefully when K < T_v.
"""

imputation_loss_type: str = "cosine"
"""
Distance metric for L_IMP.

Options:
- "cosine":  1 - cosine_similarity(Z^(final), Ẑ_ROI).
             Direction-sensitive. Scale-invariant.
             Better than MSE for high-dimensional embedding spaces.
             Recommended.
- "mse":     Mean squared error. Same as original LVR loss.
             Simple but penalises magnitude differences, not just direction.
- "nce":     Noise-contrastive estimation.
             Treats other slots in batch as negatives.
             Strongest discrimination signal but needs temperature tuning.
             τ = 0.07 is a good starting point.
"""

imputation_loss_lambda: float = 0.1
"""
λ — weight for L_IMP relative to L_NTP.

L = L_NTP + λ * L_IMP

Recommended: 0.1 (same scale as original LVR loss_lvr_lambda).
If Z becomes visually accurate but answer quality drops: reduce λ.
If Z ignores ROI content: increase λ.
Safe range: 0.05 to 0.5.
"""

nce_temperature: float = 0.07
"""
Temperature τ for NCE imputation loss.
Only used when imputation_loss_type = "nce".
"""

# ── Checkpoint and validation ──────────────────────────────────────────
checkpoint_dir: str = "checkpoints_dimv_roi"
"""
Directory for all checkpoints and validation results.
MUST be different from the existing checkpoints/ directory.
All files saved here. The existing checkpoints/ is never touched.
"""

vstar_val_fraction: float = 0.30
"""
Fraction of v*star dataset to use for validation.
The validation set is sampled ONCE at the start of training,
saved to checkpoints_dimv_roi/vstar_val_indices.json,
and reused for ALL checkpoint evaluations.
"""

validate_every_n_steps: int = 500
"""
Run validation after every N training steps.
Validation always runs after the final step regardless of this value.
"""

early_checkpoint_steps: list = None
"""
Extra checkpoint steps beyond the regular validate_every_n_steps schedule.
These are for early training diagnostics — catching collapse or divergence fast.

Default: [10, 50, 100]

At each of these steps:
  1. Save checkpoint to checkpoints_dimv_roi/checkpoint-step-{N}/
  2. Run v*star validation on the fixed 30% set
  3. Save validation result to checkpoints_dimv_roi/val_results/step_{N}.json

These early checkpoints are NOT deleted when later checkpoints arrive.
All checkpoints are kept permanently.
"""

vstar_val_seed: int = 42
"""
Random seed for sampling the 30% validation set.
Fixed so the validation set is reproducible.
"""
```

---

## Step 2 — Add ROI Pooler to latent_reasoning_module.py

**File:** `src/model/latent_reasoning_module.py`
**Action:** Add `ROIPooler` class at the top of the file, before `LatentReasoningModule`.
**Do NOT modify the existing `LatentReasoningModule` class signature** — only add
a new class and a new method.

```python
class ROIPooler(nn.Module):
    """
    Pools K ROI visual token embeddings into T_v imputation targets.

    Maps V_ROI* ∈ R^{K×d} → Ẑ_ROI ∈ R^{T_v×d}

    K varies per sample (size of bounding box in token grid, typically 4–256).
    T_v is fixed (number of reasoning slots, default 64).
    We need T_v target vectors to supervise Z^(final) slot-by-slot.

    This is a STATIC pooler — it is used only for constructing the
    imputation loss target. It does NOT appear in the inference path.
    Only Z^(final) from LatentReasoningModule is injected into the sequence.

    Three pooling strategies controlled by roi_pool_method config:
        "attention_pool": learned cross-attn, K → T_v (most expressive)
        "adaptive_pool":  PyTorch adaptive avg pool, K → T_v
        "avg_pool":       global average, K → 1 → repeat T_v times

    Args:
        d (int): Embedding dimension.
        T_v (int): Number of target slots (= number of reasoning slots).
        method (str): Pooling method. See config for options.
        num_heads (int): Attention heads for "attention_pool".
    """

    def __init__(
        self,
        d: int,
        T_v: int,
        method: str = "attention_pool",
        num_heads: int = 8,
    ):
        super().__init__()
        self.d = d
        self.T_v = T_v
        self.method = method

        if method == "attention_pool":
            # T_v learnable queries attend over K ROI tokens
            # Output: T_v pooled vectors, one per slot
            self.pool_queries = nn.Parameter(
                torch.randn(T_v, d) * (d ** -0.5)
            )
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=num_heads,
                dropout=0.0,
                batch_first=True,
            )
        # "adaptive_pool" and "avg_pool" use no learned parameters

    def forward(
        self,
        V_roi: torch.Tensor,              # [B, K, d]  ground-truth ROI embeddings
        roi_padding_mask: torch.Tensor = None,  # [B, K] True=padding
    ) -> torch.Tensor:
        """
        Pool K ROI visual embeddings into T_v slot targets.

        Args:
            V_roi: Ground-truth visual embeddings of the ROI patches.
                   Shape [B, K, d]. Extracted from V[I_ROI] for each sample.
                   K may vary across samples — padded to max K in the batch.
            roi_padding_mask: [B, K] True at padding positions (for variable K).

        Returns:
            Z_roi_target: [B, T_v, d] — imputation target for each slot.
                          This is Ẑ_ROI in the formulation.
                          L_IMP = d(Z^(final), Z_roi_target)
        """
        B = V_roi.shape[0]
        device = V_roi.device

        if self.method == "attention_pool":
            # Queries: [T_v, d] → [B, T_v, d]
            Q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
            # Cross-attend T_v queries to K ROI tokens
            pooled, _ = self.pool_attn(
                query=Q,
                key=V_roi,
                value=V_roi,
                key_padding_mask=roi_padding_mask,
            )
            # pooled: [B, T_v, d]
            return pooled

        elif self.method == "adaptive_pool":
            # Use PyTorch adaptive average pooling along the K dimension
            # V_roi: [B, K, d] → need [B, d, K] for 1D adaptive pool
            x = V_roi.transpose(1, 2)          # [B, d, K]
            # Mask out padding before pooling (set padding to 0)
            if roi_padding_mask is not None:
                # roi_padding_mask: [B, K] True=padding
                pad = roi_padding_mask.unsqueeze(1).float()  # [B, 1, K]
                x = x * (1.0 - pad)
            pooled = F.adaptive_avg_pool1d(x, self.T_v)  # [B, d, T_v]
            return pooled.transpose(1, 2)                  # [B, T_v, d]

        elif self.method == "avg_pool":
            # Global average over all K ROI tokens → scalar per d
            # Then broadcast to all T_v slots
            if roi_padding_mask is not None:
                # Mask padding tokens (set to 0 before averaging)
                valid = (~roi_padding_mask).float().unsqueeze(-1)  # [B, K, 1]
                n_valid = valid.sum(dim=1).clamp(min=1.0)          # [B, 1]
                global_avg = (V_roi * valid).sum(dim=1) / n_valid  # [B, d]
            else:
                global_avg = V_roi.mean(dim=1)  # [B, d]
            # Repeat T_v times: [B, d] → [B, T_v, d]
            return global_avg.unsqueeze(1).expand(-1, self.T_v, -1)

        else:
            raise ValueError(f"Unknown roi_pool_method: {self.method}")
```

### Also add `compute_imputation_loss` function in the same file

```python
def compute_imputation_loss(
    Z_final: torch.Tensor,         # [B, T_v, d]  model output
    Z_roi_target: torch.Tensor,    # [B, T_v, d]  pooled ROI target
    loss_type: str = "cosine",
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute L_IMP = d(Z^(final), Ẑ_ROI)

    Supervises all T_v slots simultaneously.
    This is the key upgrade over original LVR which only supervised
    the single h^lvr_start position.

    Args:
        Z_final:      [B, T_v, d]  final reasoning tokens from LatentReasoningModule
        Z_roi_target: [B, T_v, d]  pooled ROI visual embeddings from ROIPooler
        loss_type:    "cosine" | "mse" | "nce"
        temperature:  τ for NCE loss

    Returns:
        loss: scalar tensor (mean over batch and T_v slots)
    """
    B, T_v, d = Z_final.shape

    # Cast to fp32 for numerical stability (same as original LVR loss)
    Z_f = Z_final.float()
    Z_t = Z_roi_target.float()

    if loss_type == "cosine":
        # Cosine distance: 1 - cos_sim
        # Direction-sensitive, scale-invariant
        # Best for high-dimensional embedding spaces
        cos_sim = F.cosine_similarity(Z_f, Z_t, dim=-1)  # [B, T_v]
        loss = (1.0 - cos_sim).mean()

    elif loss_type == "mse":
        # Mean squared error — same family as original LVR MSE loss
        loss = F.mse_loss(Z_f, Z_t, reduction="mean")

    elif loss_type == "nce":
        # Noise-contrastive estimation
        # Positive: (Z_final[b,k], Z_roi_target[b,k]) — same slot, same sample
        # Negatives: all other (b,k) pairs in the batch

        # Reshape: [B, T_v, d] → [B*T_v, d]
        Z_f_flat = Z_f.view(B * T_v, d)
        Z_t_flat = Z_t.view(B * T_v, d)

        # Normalize to unit sphere
        Z_f_norm = F.normalize(Z_f_flat, dim=-1)   # [B*T_v, d]
        Z_t_norm = F.normalize(Z_t_flat, dim=-1)   # [B*T_v, d]

        # Similarity matrix: [B*T_v, B*T_v]
        logits = Z_f_norm @ Z_t_norm.T / temperature

        # Labels: diagonal is the positive pair
        labels = torch.arange(B * T_v, device=Z_final.device)
        loss = F.cross_entropy(logits, labels)

    else:
        raise ValueError(f"Unknown imputation_loss_type: {loss_type}")

    return loss
```

---

## Step 3 — Update Dataset Collation

**File:** `src/dataset/lvr_sft_dataset_packed.py`
**Action:** Ensure `roi_token_idxs` is preserved through collation and
accessible in the training batch. The existing `bbox_to_token_idxs()` function
already computes this — we just need to make it available to the trainer.

### 3a — Verify roi_token_idxs is in the batch

Find the collation function (likely `collate_fn` or `__getitem__`).
Ensure each batch item contains:

```python
# Each training sample must include:
{
    "input_ids":        ...,          # [seq_len] token ids
    "inputs_embeds":    ...,          # [seq_len, d] (after image injection)
    "labels":           ...,          # [seq_len] NTP labels
    "roi_token_idxs":   list[list],   # list of ROI flat indices per bounding box
                                      # e.g. [[84, 85, 86, 112, 113, 114], ...]
                                      # Inner list = indices of ROI patches in V
}
```

### 3b — Add ROI embedding extraction helper

Add this function to `data_utils.py` or `lvr_sft_dataset_packed.py`:

```python
def extract_roi_embeddings(
    image_embeds: torch.Tensor,       # [N_v, d] all visual token embeddings
    roi_token_idxs: list,             # list of int flat indices into image_embeds
    device: torch.device,
) -> torch.Tensor:
    """
    Extract ground-truth ROI visual embeddings from the vision tower output.

    This produces V_ROI* — the "ground truth for the missing variables" in
    DIMV terminology. Z^(final) must learn to match these embeddings.

    Args:
        image_embeds: [N_v, d] — all visual patch embeddings from vision tower.
                      N_v = 784 for 28×28 grid, d = 2048 for Qwen 3B.
        roi_token_idxs: Flat indices of ROI patches in the N_v token sequence.
                        Computed by bbox_to_token_idxs() from the bounding box.
        device: Target device.

    Returns:
        V_roi: [K, d] — visual embeddings of the K ROI patches.
               K = len(roi_token_idxs).
               This is the ground truth that Z must impute.
    """
    if len(roi_token_idxs) == 0:
        # No ROI annotation for this sample — return empty tensor
        return torch.zeros(0, image_embeds.shape[-1], device=device)

    idx = torch.tensor(roi_token_idxs, dtype=torch.long, device=device)
    return image_embeds[idx]  # [K, d]
```

### 3c — Collation for variable-length ROI

Since K (number of ROI tokens) varies per sample, the collator must pad V_ROI
to the maximum K in the batch:

```python
def collate_roi_embeddings(
    roi_embed_list: list,   # list of [K_i, d] tensors, one per sample
    device: torch.device,
) -> tuple:
    """
    Pad variable-length ROI embeddings to a fixed [B, K_max, d] tensor.

    Args:
        roi_embed_list: List of B tensors, each [K_i, d].
                        K_i varies per sample.

    Returns:
        V_roi_batch:    [B, K_max, d] — padded ROI embeddings.
        roi_pad_mask:   [B, K_max]    — True at padding positions.
    """
    B = len(roi_embed_list)
    d = roi_embed_list[0].shape[-1] if roi_embed_list[0].numel() > 0 else 2048
    K_max = max(max(x.shape[0] for x in roi_embed_list), 1)

    V_roi_batch = torch.zeros(B, K_max, d, device=device)
    roi_pad_mask = torch.ones(B, K_max, dtype=torch.bool, device=device)

    for b, v in enumerate(roi_embed_list):
        K_i = v.shape[0]
        if K_i > 0:
            V_roi_batch[b, :K_i] = v.to(device)
            roi_pad_mask[b, :K_i] = False  # not padding

    return V_roi_batch, roi_pad_mask
```

---

## Step 4 — Update monkey_patch_forward_lvr.py

**File:** `src/train/monkey_patch_forward_lvr.py`
**Action:** After computing Z^(final), also extract V_ROI* and store it
for the trainer to compute L_IMP.

Find the existing block:
```python
# ── LATENT REASONING MODULE: compute Z and inject ──────────────────────
```

Add the following **after** `inject_reasoning_slots` and **before** the LLM call:

```python
# ── EXTRACT ROI EMBEDDINGS FOR IMPUTATION LOSS ─────────────────────────
# This runs after image_embeds have been computed by the vision tower
# (the existing masked_scatter block has already run).
# We extract V_ROI* here — the ground truth for Z to impute.
# This is stored on the model for the trainer to access after the
# forward pass.

if (hasattr(model, 'latent_reasoning')
        and model.slot_token_ids is not None
        and hasattr(model, 'roi_pooler')
        and 'roi_token_idxs' in batch):

    roi_embed_list = []
    for b in range(B):
        # image_embeds_b: [N_v, d] — visual tokens for this sample
        # (this variable should already exist from the image injection block)
        roi_idxs = batch['roi_token_idxs'][b]
        v_roi_b = extract_roi_embeddings(
            image_embeds=image_embeds_b,
            roi_token_idxs=roi_idxs,
            device=inputs_embeds.device,
        )
        roi_embed_list.append(v_roi_b)

    # Pad to [B, K_max, d]
    V_roi_batch, roi_pad_mask = collate_roi_embeddings(
        roi_embed_list=roi_embed_list,
        device=inputs_embeds.device,
    )

    # Pool K → T_v to get per-slot imputation targets
    # Z_roi_target: [B, T_v, d]
    Z_roi_target = model.roi_pooler(
        V_roi=V_roi_batch,
        roi_padding_mask=roi_pad_mask,
    )

    # Store for trainer access
    # Both Z_final and Z_roi_target are needed to compute L_IMP
    model._last_Z_final = Z_final          # [B, T_v, d] — model output
    model._last_Z_roi_target = Z_roi_target  # [B, T_v, d] — pooled ROI target
# ── END ROI EXTRACTION ──────────────────────────────────────────────────
```

---

## Step 5 — Update qwen_lvr_model.py

**File:** `src/model/qwen_lvr_model.py`
**Action:** Register `ROIPooler` alongside the existing `LatentReasoningModule`.

In `QwenWithLVR.__init__`, add after the existing `latent_reasoning` block:

```python
# ── ROI Pooler (for imputation loss supervision) ──────────────────────
if (hasattr(config, 'latent_reasoning_config')
        and config.latent_reasoning_config is not None
        and config.latent_reasoning_config.use_roi_supervision):

    lr_cfg = config.latent_reasoning_config

    self.roi_pooler = ROIPooler(
        d=config.hidden_size,
        T_v=lr_cfg.num_reasoning_slots,
        method=lr_cfg.roi_pool_method,
        num_heads=lr_cfg.num_attn_heads,
    )

    # Storage for trainer access
    self._last_Z_final = None        # [B, T_v, d]
    self._last_Z_roi_target = None   # [B, T_v, d]

    print(f"[DIMV-ROI] ROIPooler method={lr_cfg.roi_pool_method}, "
          f"T_v={lr_cfg.num_reasoning_slots}, "
          f"lambda={lr_cfg.imputation_loss_lambda}")
```

### Add to imports at top of file

```python
from src.model.latent_reasoning_module import (
    LatentReasoningModule,
    ROIPooler,
    compute_imputation_loss,
)
```

---

## Step 6 — Update lvr_trainer.py

**File:** `src/trainer/lvr_trainer.py`
**Action:** Add L_IMP to the loss, add checkpoint saving to `checkpoints_dimv_roi/`,
add v*star validation after each checkpoint.

### 6a — Loss computation

Find the existing loss computation block. **Replace**:
```python
loss = outputs.loss  # NTP only
```

**With**:
```python
# ── NTP Loss ─────────────────────────────────────────────────────────
loss_ntp = outputs.loss

# ── Imputation Loss (L_IMP) ───────────────────────────────────────────
loss_imp = torch.tensor(0.0, device=loss_ntp.device)
if (hasattr(self.model, 'roi_pooler')
        and self.model._last_Z_final is not None
        and self.model._last_Z_roi_target is not None
        and self.args.use_roi_supervision):

    loss_imp = compute_imputation_loss(
        Z_final=self.model._last_Z_final,
        Z_roi_target=self.model._last_Z_roi_target,
        loss_type=self.args.imputation_loss_type,
        temperature=self.args.nce_temperature,
    )

# ── Total Loss ────────────────────────────────────────────────────────
lam = getattr(self.args, 'imputation_loss_lambda', 0.1)
loss = loss_ntp + lam * loss_imp

# ── Logging ───────────────────────────────────────────────────────────
self.log("train/loss",     loss.item())
self.log("train/loss_ntp", loss_ntp.item())
self.log("train/loss_imp", loss_imp.item())
```

### 6b — Checkpoint saving

Find the existing `_save_checkpoint` method or the training step where checkpoints
are saved. **Add** the following logic:

```python
def _save_checkpoint_dimv_roi(self, step: int) -> str:
    """
    Save checkpoint to checkpoints_dimv_roi/ folder.

    NEVER deletes existing checkpoints.
    NEVER touches the existing checkpoints/ folder.
    Each checkpoint is saved to a new step-specific subdirectory.

    Returns:
        Path to saved checkpoint directory.
    """
    import os
    checkpoint_dir = getattr(self.args, 'checkpoint_dir', 'checkpoints_dimv_roi')
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-step-{step}")

    # Create directory if it does not exist
    # exist_ok=True ensures we never crash if called twice
    os.makedirs(ckpt_path, exist_ok=True)

    # Verify we are NOT writing to the original checkpoints/ folder
    original_ckpt_dir = "checkpoints"
    assert not ckpt_path.startswith(original_ckpt_dir + "/"), (
        f"SAFETY CHECK FAILED: attempting to write to {ckpt_path}. "
        f"This would touch the original checkpoints/ folder. Aborting."
    )

    # Save model state (LoRA adapters + new modules)
    self.model.save_pretrained(ckpt_path)
    self.tokenizer.save_pretrained(ckpt_path)

    # Save training state
    import json
    state = {
        "step": step,
        "loss_ntp": float(self._last_logged_loss_ntp),
        "loss_imp": float(self._last_logged_loss_imp),
        "loss_total": float(self._last_logged_loss),
    }
    with open(os.path.join(ckpt_path, "train_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"[DIMV-ROI] Checkpoint saved: {ckpt_path}")
    return ckpt_path
```

### 6c — Post-checkpoint validation

```python
def _validate_on_vstar(self, step: int, ckpt_path: str) -> dict:
    """
    Run validation on the fixed 30% v*star sample after saving a checkpoint.

    The validation set is loaded from checkpoints_dimv_roi/vstar_val_indices.json.
    This file is created once at training start and reused for all checkpoints.
    Using the same set for all checkpoints ensures fair comparison.

    Saves a JSON file to checkpoints_dimv_roi/val_results/step_{step}.json
    containing:
        - summary metrics (accuracy, n_correct, n_samples)
        - per-sample results (idx, question, pred, gold, correct)
        - training state at validation time (step, loss values)
        - checkpoint path

    Args:
        step:      Current training step.
        ckpt_path: Path to the checkpoint just saved.

    Returns:
        metrics: dict with summary validation metrics.
    """
    import os, json
    from eval.vstar_validator import VStarValidator

    checkpoint_dir = getattr(self.args, 'checkpoint_dir', 'checkpoints_dimv_roi')
    val_indices_path = os.path.join(checkpoint_dir, "vstar_val_indices.json")
    val_results_dir = os.path.join(checkpoint_dir, "val_results")
    os.makedirs(val_results_dir, exist_ok=True)

    # Load fixed validation set indices
    with open(val_indices_path, "r") as f:
        val_indices = json.load(f)

    print(f"[DIMV-ROI] Running v*star validation: "
          f"{len(val_indices)} samples at step {step} ...")

    # Run evaluation
    validator = VStarValidator(
        model=self.model,
        tokenizer=self.tokenizer,
        val_indices=val_indices,
        device=self.args.device,
    )
    metrics = validator.evaluate()

    # Build full result record
    result_record = {
        # ── Identity ────────────────────────────────────────────────
        "step":             step,
        "checkpoint_path":  ckpt_path,

        # ── Training state at this step ──────────────────────────────
        "train_loss":       getattr(self, '_last_logged_loss',     None),
        "train_loss_ntp":   getattr(self, '_last_logged_loss_ntp', None),
        "train_loss_imp":   getattr(self, '_last_logged_loss_imp', None),

        # ── Validation summary ───────────────────────────────────────
        "val_accuracy":     metrics["accuracy"],
        "val_n_correct":    metrics["n_correct"],
        "val_n_samples":    metrics["n_samples"],

        # ── Per-sample details ───────────────────────────────────────
        # Each entry: {idx, question, pred, gold, correct}
        # Useful for error analysis — which questions improved/regressed
        "per_sample":       metrics["per_sample"],
    }

    # Save JSON — one file per step, never overwrite
    result_path = os.path.join(val_results_dir, f"step_{step}.json")
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2, ensure_ascii=False)

    # Log summary metrics to wandb/tensorboard
    self.log("val/accuracy",   metrics["accuracy"])
    self.log("val/n_correct",  metrics["n_correct"])
    self.log("val/n_samples",  metrics["n_samples"])

    print(f"[DIMV-ROI] Step {step} | "
          f"val_accuracy={metrics['accuracy']:.4f} "
          f"({metrics['n_correct']}/{metrics['n_samples']})")
    print(f"[DIMV-ROI] Results saved → {result_path}")

    return metrics
```

### 6d — Checkpoint and validation schedule

Replace the existing checkpoint hook with this complete schedule logic.
This handles early steps (10, 50, 100), regular steps (every 500),
and the final step — all saving to `checkpoints_dimv_roi/` and running
v*star validation after every save.

```python
def _should_checkpoint(self, step: int) -> bool:
    """
    Returns True if a checkpoint should be saved at this step.

    Checkpoint schedule:
      Early steps:   10, 50, 100              (for fast divergence detection)
      Regular steps: every validate_every_n_steps (default 500)
      Final step:    args.max_steps            (always)
    """
    early_steps = getattr(self.args, 'early_checkpoint_steps', [10, 50, 100])
    if isinstance(early_steps, str):
        early_steps = [int(s) for s in early_steps.split(',')]

    validate_every = getattr(self.args, 'validate_every_n_steps', 500)
    max_steps = self.args.max_steps

    return (
        step in early_steps
        or step % validate_every == 0
        or step == max_steps
    )


# In training_step() or on_step_end(), add:
if self._should_checkpoint(current_step):
    ckpt_path = self._save_checkpoint_dimv_roi(current_step)
    self._validate_on_vstar(current_step, ckpt_path)
```

---

## Step 7 — Create v*star Validator

**File:** `src/eval/vstar_validator.py`
**Action:** Create from scratch.

```python
"""
V*star fixed-set validator for DIMV-ROI training.

The validation set is 30% of the v*star benchmark, randomly sampled ONCE
at training start using seed 42. The same indices are reused for all
checkpoint evaluations to ensure fair comparison across steps.
"""

import os
import json
import random
import torch
from typing import Optional


class VStarValidator:
    """
    Evaluates model on a fixed subset of the v*star benchmark.

    Usage:
        # At training start — create and save the fixed validation set ONCE:
        VStarValidator.create_fixed_val_set(
            vstar_dataset=dataset,
            val_fraction=0.30,
            seed=42,
            save_path="checkpoints_dimv_roi/vstar_val_indices.json",
        )

        # At each checkpoint — load and evaluate:
        validator = VStarValidator(model, tokenizer, val_indices, device)
        metrics = validator.evaluate()
    """

    def __init__(
        self,
        model,
        tokenizer,
        val_indices: list,
        device: torch.device,
        batch_size: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_indices = val_indices
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def create_fixed_val_set(
        vstar_dataset,
        val_fraction: float = 0.30,
        seed: int = 42,
        save_path: str = "checkpoints_dimv_roi/vstar_val_indices.json",
    ) -> list:
        """
        Sample val_fraction of v*star once and save indices to disk.

        CALL THIS ONCE AT THE START OF TRAINING.
        If the file already exists, load from it instead (idempotent).

        Args:
            vstar_dataset: The full v*star dataset object.
            val_fraction:  Fraction to use as validation (0.30 = 30%).
            seed:          Random seed for reproducibility.
            save_path:     Where to save the index list as JSON.

        Returns:
            val_indices: list of int indices into vstar_dataset.
        """
        # If already exists, load and return (do not re-sample)
        if os.path.exists(save_path):
            print(f"[VStarValidator] Loading existing validation set: {save_path}")
            with open(save_path, "r") as f:
                val_indices = json.load(f)
            print(f"[VStarValidator] Loaded {len(val_indices)} validation samples.")
            return val_indices

        # Sample once with fixed seed
        total = len(vstar_dataset)
        n_val = int(total * val_fraction)
        rng = random.Random(seed)
        all_indices = list(range(total))
        val_indices = rng.sample(all_indices, n_val)
        val_indices.sort()  # sort for reproducibility of order

        # Save to disk
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(val_indices, f, indent=2)

        print(f"[VStarValidator] Created validation set: "
              f"{len(val_indices)}/{total} samples ({val_fraction*100:.0f}%).")
        print(f"[VStarValidator] Saved to: {save_path}")
        return val_indices

    def evaluate(self) -> dict:
        """
        Run evaluation on the fixed validation set.

        Returns:
            metrics: dict containing:
                - "accuracy":    exact match accuracy on answer tokens
                - "n_correct":   number of correct predictions
                - "n_samples":   number of evaluated samples
                - "per_sample":  list of per-sample results (for debugging)
        """
        self.model.eval()
        correct = 0
        total = 0
        per_sample_results = []

        with torch.no_grad():
            for i in range(0, len(self.val_indices), self.batch_size):
                batch_indices = self.val_indices[i : i + self.batch_size]
                batch_results = self._eval_batch(batch_indices)
                for r in batch_results:
                    per_sample_results.append(r)
                    if r["correct"]:
                        correct += 1
                    total += 1

        self.model.train()

        accuracy = correct / total if total > 0 else 0.0
        metrics = {
            "accuracy":   round(accuracy, 4),
            "n_correct":  correct,
            "n_samples":  total,
            "per_sample": per_sample_results,
        }
        return metrics

    def _eval_batch(self, indices: list) -> list:
        """
        Evaluate one batch. Returns list of per-sample result dicts.

        Each result dict has:
            {
                "idx":        int,    sample index in the full dataset
                "question":   str,    the question text
                "pred":       str,    model's predicted answer
                "gold":       str,    ground-truth answer
                "correct":    bool,   whether pred matches gold (exact match)
            }

        NOTE TO CLAUDE CODE:
        Implement this method based on how the v*star dataset is structured.
        General pattern:
            1. Load samples at `indices` from the v*star dataset
            2. Tokenize each (image, question) pair
            3. Run model.generate() for each sample
            4. Decode generated tokens to string
            5. Compare to ground truth answer (exact match or normalised)
            6. Return list of result dicts as described above

        The model already has the attention mask and Z injection from the
        training setup — inference uses the same forward pass.
        """
        raise NotImplementedError(
            "Implement _eval_batch() based on v*star dataset format. "
            "Return a list of dicts: "
            "[{'idx': int, 'question': str, 'pred': str, 'gold': str, 'correct': bool}, ...]"
        )
```

---

## Step 8 — Training Startup: Create Fixed Val Set

**File:** `train-stage1-3b-dimv-roi.sh` (new training script)
**Action:** Create based on existing training script. Add ROI supervision flags.

```bash
#!/bin/bash
# DIMV-ROI training: DIMV-like latent reasoning with ROI supervision
# Saves checkpoints to checkpoints_dimv_roi/ — does NOT touch checkpoints/

python train.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    \
    # ── Existing flags (keep unchanged) ──────────────────────────────
    --lora_rank 64 \
    --lora_alpha 128 \
    --learning_rate 2e-4 \
    --max_steps 2500 \
    --freeze_vision_tower True \
    --latent_reasoning True \
    --num_reasoning_slots 64 \
    --num_refinement_steps 2 \
    --slot_init "learned" \
    --use_layer_norm True \
    --use_ffn True \
    \
    # ── New ROI supervision flags ─────────────────────────────────────
    --use_roi_supervision True \
    --roi_pool_method "attention_pool" \
    --imputation_loss_type "cosine" \
    --imputation_loss_lambda 0.1 \
    \
    # ── Checkpoint and validation flags ──────────────────────────────
    --checkpoint_dir "checkpoints_dimv_roi" \
    --validate_every_n_steps 500 \
    --early_checkpoint_steps "10,50,100" \
    --vstar_val_fraction 0.30 \
    --vstar_val_seed 42 \
    \
    # ── Learning rates ────────────────────────────────────────────────
    --latent_reasoning_lr 5e-4 \
    --roi_pooler_lr 5e-4
```

### Add to training main() before training loop starts

```python
# In train.py or trainer setup — add this BEFORE the training loop:

from eval.vstar_validator import VStarValidator

# Create the fixed validation set ONCE (idempotent — safe to call every run)
# If vstar_val_indices.json already exists, this just loads it.
val_indices = VStarValidator.create_fixed_val_set(
    vstar_dataset=vstar_dataset,           # your v*star dataset object
    val_fraction=args.vstar_val_fraction,  # 0.30
    seed=args.vstar_val_seed,              # 42
    save_path=os.path.join(
        args.checkpoint_dir,
        "vstar_val_indices.json"
    ),
)
print(f"[DIMV-ROI] Fixed validation set: {len(val_indices)} samples. "
      f"Reusing same set for all checkpoints.")
```

---

## Step 9 — Parameter Groups

**File:** `src/trainer/lvr_trainer.py`
**Action:** Add `roi_pooler` to the high-LR parameter group.

```python
# Find the existing param_groups definition and ADD roi_pooler:
param_groups = [
    {
        "params": [p for n, p in model.named_parameters()
                   if ("latent_reasoning" in n or "roi_pooler" in n)
                   and p.requires_grad],
        "lr": args.latent_reasoning_lr,   # 5e-4
        "name": "latent_reasoning_and_roi_pooler",
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if "lora" in n.lower() and p.requires_grad],
        "lr": args.learning_rate,          # 2e-4
        "name": "lora",
    },
]
```

---

## Frozen vs Trainable Parameters

| Component | Status | Notes |
|-----------|--------|-------|
| ViT vision tower | **Frozen** | Same as existing |
| Visual merger | **Frozen** | Same as existing |
| Qwen LLM base | **Frozen** | Same as existing |
| Qwen LoRA (rank=64) | **Trained** | LR = 2e-4 |
| `slot_queries` in LatentReasoningModule | **Trained** | LR = 5e-4 |
| `cross_attn` in LatentReasoningModule | **Trained** | LR = 5e-4 |
| `context_proj` in LatentReasoningModule | **Trained** | LR = 5e-4 |
| `ffn`, `norm1`, `norm2` in LatentReasoningModule | **Trained** | LR = 5e-4 |
| `pool_queries` in ROIPooler | **Trained** | LR = 5e-4 (only for attention_pool) |
| `pool_attn` in ROIPooler | **Trained** | LR = 5e-4 (only for attention_pool) |

---

## Checkpoint Directory Structure

```
checkpoints_dimv_roi/                    ← ALL new files go here
├── vstar_val_indices.json               ← Fixed 30% validation set (created ONCE)
│
├── checkpoint-step-10/                  ← Early: fast divergence check
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   ├── train_state.json
│   └── tokenizer files
├── checkpoint-step-50/                  ← Early: slot collapse check
│   └── ...
├── checkpoint-step-100/                 ← Early: loss trend confirmed
│   └── ...
├── checkpoint-step-500/                 ← Regular
│   └── ...
├── checkpoint-step-1000/
│   └── ...
├── checkpoint-step-1500/
│   └── ...
├── checkpoint-step-2000/
│   └── ...
├── checkpoint-step-2500/                ← Final
│   └── ...
│
└── val_results/
    ├── step_10.json                     ← Early validation result
    ├── step_50.json
    ├── step_100.json
    ├── step_500.json
    ├── step_1000.json
    ├── step_1500.json
    ├── step_2000.json
    └── step_2500.json

checkpoints/                             ← EXISTING — READ ONLY — NEVER TOUCH
├── ...existing checkpoints...
```

### JSON result file format

Every `val_results/step_N.json` file has this structure:

```json
{
  "step": 100,
  "checkpoint_path": "checkpoints_dimv_roi/checkpoint-step-100",
  "train_loss":     0.8312,
  "train_loss_ntp": 0.7891,
  "train_loss_imp": 0.4211,
  "val_accuracy":   0.4120,
  "val_n_correct":  123,
  "val_n_samples":  299,
  "per_sample": [
    {
      "idx":      42,
      "question": "What is the abnormality in this region?",
      "pred":     "necrosis",
      "gold":     "necrosis",
      "correct":  true
    },
    {
      "idx":      87,
      "question": "Is there glomerular sclerosis?",
      "pred":     "no",
      "gold":     "yes",
      "correct":  false
    }
  ]
}
```

---

## Monitoring Checklist

| Step | What to check | Action if wrong |
|------|--------------|-----------------|
| **10** | `train/loss` not NaN or Inf | If NaN: lower LR, check ROI extraction |
| **10** | `train/loss_imp` < 1.0 | If >> 1.0: check pooler output shapes |
| **50** | `train/loss` decreasing | If flat: check mask, check gradient flow |
| **50** | `val/accuracy` > random baseline | If at random: Z not helping — check injection |
| **100** | `train/loss_imp` clearly dropping | ROI supervision must be active by step 100 |
| **100** | Slot attention entropy < 6.0 | Slots must start focusing; if still at max: check X_o |
| **500+** | `val/accuracy` improving each checkpoint | Compare step_N.json files |
| **500+** | `train/loss_ntp` << `train/loss_imp` | NTP should dominate; if reversed: reduce λ |
| **2500** | `val/accuracy` highest at final step | If peak was earlier: overfitting to ROI |

---

## Ablation Table (for comparison)

| Model | L_NTP | L_IMP | ROI supervision | Notes |
|-------|-------|-------|-----------------|-------|
| DIMV baseline | ✓ | ✗ | ✗ | Existing implementation |
| DIMV-ROI (this) | ✓ | ✓ | ✓ | New implementation |
| DIMV-ROI ablation | ✓ | ✗ | ✗ | Set use_roi_supervision=False |

Run all three to measure the contribution of L_IMP.

---

## Utility: Compare Validation Results Across Checkpoints

**File:** `src/eval/compare_val_results.py`
**Action:** Create from scratch. Run this after training to compare all checkpoints.

```python
"""
Load and compare all validation result JSON files across checkpoints.

Usage:
    python src/eval/compare_val_results.py \
        --results_dir checkpoints_dimv_roi/val_results
"""

import os
import json
import argparse


def load_all_results(results_dir: str) -> list:
    """
    Load all step_N.json files from results_dir.
    Returns list of result dicts sorted by step number.
    """
    records = []
    for fname in os.listdir(results_dir):
        if not fname.startswith("step_") or not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r") as f:
            record = json.load(f)
        records.append(record)

    # Sort by step number
    records.sort(key=lambda r: r["step"])
    return records


def print_summary_table(records: list):
    """
    Print a summary table of validation accuracy across all checkpoints.

    Example output:
        Step  | Val Acc | Correct/Total | NTP Loss | IMP Loss
        ------+---------+---------------+----------+---------
            10 |  0.2341 |    70 / 299   |   1.8231 |   0.7812
            50 |  0.3010 |    90 / 299   |   1.2310 |   0.5231
           100 |  0.3679 |   110 / 299   |   0.9812 |   0.3901
           500 |  0.4120 |   123 / 299   |   0.8312 |   0.2341
    """
    header = (
        f"{'Step':>6} | {'Val Acc':>8} | {'Correct/Total':>14} "
        f"| {'NTP Loss':>9} | {'IMP Loss':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    best_acc = -1
    best_step = -1

    for r in records:
        step     = r.get("step", "?")
        acc      = r.get("val_accuracy", 0.0)
        correct  = r.get("val_n_correct", 0)
        total    = r.get("val_n_samples", 0)
        loss_ntp = r.get("train_loss_ntp") or 0.0
        loss_imp = r.get("train_loss_imp") or 0.0

        marker = " ← best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_step = step

        print(
            f"{step:>6} | {acc:>8.4f} | {correct:>6} / {total:<6} "
            f"| {loss_ntp:>9.4f} | {loss_imp:>9.4f}{marker}"
        )

    print(sep)
    print(f"Best checkpoint: step {best_step} with val_accuracy = {best_acc:.4f}")
    print(sep)


def analyse_errors(records: list, step_a: int, step_b: int):
    """
    Compare per-sample results between two checkpoints.
    Shows which samples improved (wrong→correct) and regressed (correct→wrong).

    Args:
        records: All loaded result records.
        step_a:  Earlier step number.
        step_b:  Later step number (should be higher than step_a).
    """
    rec_a = next((r for r in records if r["step"] == step_a), None)
    rec_b = next((r for r in records if r["step"] == step_b), None)

    if rec_a is None or rec_b is None:
        print(f"Could not find results for steps {step_a} and/or {step_b}.")
        return

    # Build idx → result maps
    map_a = {s["idx"]: s for s in rec_a.get("per_sample", [])}
    map_b = {s["idx"]: s for s in rec_b.get("per_sample", [])}

    improved  = []  # wrong at step_a, correct at step_b
    regressed = []  # correct at step_a, wrong at step_b

    for idx in set(map_a.keys()) & set(map_b.keys()):
        was_correct = map_a[idx]["correct"]
        now_correct = map_b[idx]["correct"]
        if not was_correct and now_correct:
            improved.append(map_b[idx])
        elif was_correct and not now_correct:
            regressed.append(map_b[idx])

    print(f"\nStep {step_a} → Step {step_b}:")
    print(f"  Improved (wrong→correct): {len(improved)}")
    print(f"  Regressed (correct→wrong): {len(regressed)}")

    if improved:
        print("\n  Sample improved predictions:")
        for s in improved[:5]:  # show up to 5
            print(f"    idx={s['idx']} | Q: {s['question'][:60]} "
                  f"| pred={s['pred']} | gold={s['gold']}")

    if regressed:
        print("\n  Sample regressions:")
        for s in regressed[:5]:
            print(f"    idx={s['idx']} | Q: {s['question'][:60]} "
                  f"| pred={s['pred']} | gold={s['gold']}")


def save_summary_json(records: list, output_path: str):
    """
    Save a compact summary JSON of all checkpoints for programmatic use.

    Output format:
    [
      {"step": 10,  "val_accuracy": 0.2341, "n_correct": 70,  "n_samples": 299, ...},
      {"step": 50,  "val_accuracy": 0.3010, "n_correct": 90,  "n_samples": 299, ...},
      ...
    ]
    """
    summary = []
    for r in records:
        summary.append({
            "step":          r.get("step"),
            "val_accuracy":  r.get("val_accuracy"),
            "val_n_correct": r.get("val_n_correct"),
            "val_n_samples": r.get("val_n_samples"),
            "train_loss":    r.get("train_loss"),
            "train_loss_ntp":r.get("train_loss_ntp"),
            "train_loss_imp":r.get("train_loss_imp"),
            "checkpoint_path": r.get("checkpoint_path"),
        })

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="checkpoints_dimv_roi/val_results",
        help="Path to val_results/ directory",
    )
    parser.add_argument(
        "--compare_steps",
        default=None,
        help="Compare two specific steps, e.g. --compare_steps 100,2500",
    )
    parser.add_argument(
        "--save_summary",
        default="checkpoints_dimv_roi/val_summary.json",
        help="Where to save the compact summary JSON",
    )
    args = parser.parse_args()

    records = load_all_results(args.results_dir)
    if not records:
        print(f"No result files found in {args.results_dir}")
        exit(1)

    print(f"\nFound {len(records)} checkpoint result(s).\n")
    print_summary_table(records)

    if args.compare_steps:
        step_a, step_b = [int(s) for s in args.compare_steps.split(",")]
        analyse_errors(records, step_a, step_b)

    save_summary_json(records, args.save_summary)
``` 