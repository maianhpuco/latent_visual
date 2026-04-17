# Prototype-Based Parallel LVR: Implementation Guide for Claude Code

## Context & Goal

This document guides the modification of an existing LVR (Latent Visual Reasoning)
codebase based on Qwen2.5-VL-3B to replace sequential autoregressive latent token
generation with **parallel prototype-based inference**.

### The Problem Being Solved

The original LVR generates K reasoning tokens sequentially:
```
z_1 → z_2 → ... → z_K   (each depends on the previous)
```
This causes error accumulation: if z_1 is wrong, all subsequent tokens are corrupted.
Error variance grows as E[‖ε_K‖²] ≥ K·σ².

### The Solution

Replace sequential generation with K **independent prototype slots**:
```
z_k = g_φ(p_k, O)   for all k simultaneously
```
Where:
- `p_k` ∈ R^D is a **learnable prototype query** (nn.Parameter) for slot k
- `O` = concat(V, C) is the observed context (image embeddings + text embeddings)
- `g_φ` is a cross-attention module — no z_j appears in the computation of z_k
- All K outputs are computed in **one forward pass, no sequential dependency**

Error variance stays at σ² regardless of K (proven: ∂z_k/∂z_j = 0 for k≠j).

---

## Notation Reference

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `I` | H×W×3 | Input image |
| `q` | string | Natural language question |
| `y` | string | Ground-truth answer tokens |
| `V` | [N_v, D] | Visual token embeddings from ViT. N_v=784 for 28×28 grid, D=2048 (Qwen 3B) |
| `C` | [N_c, D] | Text context embeddings (question tokens) |
| `O` | [N_v+N_c, D] | Observed context = concat(V, C). Never contains any z_k |
| `K` | scalar | Number of prototype slots. Hyperparameter, default=8 |
| `D` | scalar | Embedding dimension = LLM hidden size = 2048 for Qwen 3B |
| `P` | [K, D] | Learnable prototype bank = [p_1,...,p_K]. `nn.Parameter`, trainable |
| `p_k` | [D] | The k-th prototype query. Encodes "what aspect am I looking for?" |
| `z_k` | [D] | The k-th inferred prototype — actual extracted information for slot k |
| `Z` | [K, D] | Full prototype matrix = [z_1,...,z_K]. Injected into LLM sequence |
| `A` | [K, N_v+N_c] | Attention weights. A[k,n] = how much slot k uses context token n |
| `g_φ` | module | Prototype cross-attention: (P, O) → Z |
| `f_θ` | module | Qwen 3B LLM (frozen base + LoRA adapters) |
| `λ_1` | scalar | Weight for diversity loss. Default=0.05 |
| `λ_2` | scalar | Weight for focus/entropy loss. Default=0.01 |

---

## Files to Create or Modify

```
src/
├── model/
│   ├── lvr_heads.py                   ← ADD PrototypeBank and PrototypeCrossAttention
│   └── qwen_lvr_model.py              ← MODIFY: register new modules, update forward
├── train/
│   └── monkey_patch_forward_lvr.py    ← MODIFY: replace sequential injection with parallel
├── trainer/
│   └── lvr_trainer.py                 ← MODIFY: update loss computation
├── dataset/
│   └── lvr_sft_dataset_packed.py      ← MODIFY: add [proto] tokens to vocabulary handling
└── config/
    └── prototype_lvr_config.py        ← CREATE: new config dataclass
```

---

## Step 1: New Config Dataclass

**File:** `src/config/prototype_lvr_config.py`

**Create this file from scratch.**

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrototypeLVRConfig:
    """
    Configuration for prototype-based parallel LVR.
    Replaces sequential [lvr] token generation with K parallel prototype slots.

    Key design decisions:
    - K prototype slots inferred independently, no slot-to-slot dependency
    - P (prototype bank) is nn.Parameter, NOT produced by any sub-network
    - Diversity loss prevents all slots collapsing to the same vector
    - Focus loss encourages each slot to attend to specific image patches
    """

    # ── Prototype architecture ─────────────────────────────────────────────
    num_prototypes: int = 8
    """
    K = number of prototype slots.
    Each slot learns to represent a different aspect of the image:
    one for object identity, one for texture, one for context, etc.
    These specialties emerge from training — not from manual design.
    Start with K=4, increase if loss_CE plateaus.
    """

    prototype_dim: int = 2048
    """
    D = dimension of each prototype vector.
    Must equal Qwen 3B hidden size (2048). Do not change.
    """

    prototype_num_heads: int = 8
    """
    Number of attention heads in the prototype cross-attention module.
    Each head can attend to a different set of patches within one slot.
    Must divide prototype_dim evenly: 2048 / 8 = 256 per head.
    """

    prototype_dropout: float = 0.1
    """Dropout in cross-attention. Acts as alpha-regularisation (analog of DIMV's αI)."""

    prototype_init_scale: float = None
    """
    Initialisation scale for P. If None, uses D^{-0.5} = 1/sqrt(2048) ≈ 0.022.
    This matches the √D denominator in attention score computation to prevent
    initial attention from being saturated or uniform.
    """

    # ── Loss weights ───────────────────────────────────────────────────────
    loss_diversity_lambda: float = 0.05
    """
    λ_1 — weight for prototype diversity loss L_div.
    L_div = mean squared cosine similarity between all prototype pairs.
    L_div → 0 means slots are orthogonal (maximally diverse).
    L_div → 1 means slot collapse (all slots identical — bad).

    If loss_div > 0.8 after 500 steps: increase this.
    If loss_CE diverges: decrease this.
    Safe range: 0.01 to 0.1.
    """

    loss_focus_lambda: float = 0.01
    """
    λ_2 — weight for attention entropy loss L_focus.
    L_focus = mean attention entropy across all K slots.
    Low entropy = each slot focuses on specific image patches (compact, good).
    High entropy = each slot spreads weight uniformly (diffuse, bad).

    Analog of DIMV's feature selection threshold τ.
    Safe range: 0.005 to 0.05.
    """

    # ── Training schedule ──────────────────────────────────────────────────
    warmup_steps_prototype_only: int = 500
    """
    Steps 0→warmup: freeze LoRA, train only P and g_φ.
    Lets the prototype module find useful signal before diversity pressure.
    """

    diversity_anneal_start: int = 500
    diversity_anneal_end: int = 1500
    """
    Steps over which λ_1 is annealed from 0 to loss_diversity_lambda.
    Prevents diversity loss from disrupting early learning.
    """

    # ── Sequence integration ───────────────────────────────────────────────
    proto_token_id_start: int = None
    """
    Token ID of [proto_1] in the vocabulary. Subsequent proto tokens are
    proto_token_id_start + k for k in {0,...,K-1}.
    Set automatically by model init from the tokenizer.
    """

    # ── Compatibility with original LVR ───────────────────────────────────
    keep_lvr_loss: bool = False
    """
    If True, also compute the original LVR MSE loss (h^lvr_start vs v*).
    Useful for ablation: compare with and without the original signal.
    Default False — prototype loss replaces it entirely.
    """
```

---

## Step 2: New Modules in lvr_heads.py

**File:** `src/model/lvr_heads.py`

**Add the following two classes after the existing `LVRHead` and `LVRHeadGLU` classes.**
Do not modify the existing classes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """
    Learnable prototype slot bank P = [p_1, p_2, ..., p_K] ∈ R^{K×D}.

    P is a raw nn.Parameter — NOT produced by any sub-network.
    It is the same for every input sample: only Z = g_φ(P, O) varies
    per sample because O is image-specific.

    After training, each p_k has learned a different "aspect query":
    - p_1 might ask "what is the main object?"
    - p_2 might ask "what is the local texture?"
    - p_3 might ask "what is the spatial layout?"
    These specialties are NOT programmed — they emerge from L_CE + L_div.

    Args:
        K (int): Number of prototype slots.
        D (int): Embedding dimension (must equal LLM hidden size = 2048).
        init_scale (float): Initialisation std. Defaults to D^{-0.5}.
    """

    def __init__(self, K: int, D: int, init_scale: float = None):
        super().__init__()
        self.K = K
        self.D = D
        scale = init_scale if init_scale is not None else D ** -0.5

        # P: the core learnable parameter. Shape [K, D].
        # Every training step, gradient ∂L/∂P updates all K rows.
        self.P = nn.Parameter(torch.randn(K, D) * scale)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            P: shape [K, D] — the current prototype bank.
        """
        return self.P


class PrototypeCrossAttention(nn.Module):
    """
    Prototype cross-attention module g_φ: (P, O) → Z.

    Mathematical formulation:
        A   = softmax(P · O^T / √D)   shape [K, N_obs]
        Z̃  = A · O                    shape [K, D]
        Z   = Proj(LN(Z̃))             shape [K, D]

    KEY PROPERTY — independence guaranteed by architecture:
        z_k = g_φ(p_k, O)  depends ONLY on p_k and O.
        ∂z_k / ∂z_j = 0 for all j ≠ k.
        No z_j ever appears in the computation of z_k.

    This eliminates the error cascade of sequential generation:
        Sequential: E[‖ε_K‖²] ≥ K·σ²   (grows with K)
        Parallel:   E[‖ε_k‖²]  = σ²     (constant for all k)

    Connection to DIMV:
        A plays the role of Σ_mo^T (Σ_o + αI)^{-1}:
        - Σ_mo in DIMV: covariance between missing and observed features
          (which observed features predict which missing feature)
        - A here: learned attention weights encoding the same relationship
          but nonlinearly and per-image rather than from Gaussian statistics.
        - dropout in cross-attention plays the role of α regularisation.

    Args:
        D (int): Embedding dimension = LLM hidden size.
        num_heads (int): Attention heads. Must divide D evenly.
        dropout (float): Attention dropout.
    """

    def __init__(self, D: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.D = D

        # Cross-attention: Q = prototype queries P (broadcast across batch)
        #                  K = V = observed context O
        # batch_first=True: tensors are [B, seq_len, D]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Context normalisation: analog of (Σ_o + αI) regularisation in DIMV.
        # Stabilises attention computation, prevents extreme dot products.
        self.context_norm = nn.LayerNorm(D)

        # Projection head: maps attended features → prototype space.
        # Shared across all K slots (same W_1, W_2 applied to each row).
        # Expands D → 2D → D to allow nonlinear transformation.
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )

    def forward(
        self,
        P: torch.Tensor,                      # [K, D] prototype bank
        O: torch.Tensor,                      # [B, N_obs, D] observed context
        key_padding_mask: torch.Tensor = None, # [B, N_obs] True=padding position
    ):
        """
        Infer all K prototype vectors in one parallel forward pass.

        Args:
            P: Prototype bank from PrototypeBank.forward(). Shape [K, D].
               Same for all batch items — will be broadcast across B.
            O: Observed context = concat(V_masked, C). Shape [B, N_obs, D].
               IMPORTANT: O must NOT contain any z_k values.
               This enforces the DIMV principle: condition only on truly
               observed quantities. Without this, the model can trivially
               copy ground-truth embeddings (shortcut learning).
            key_padding_mask: True at padding positions. Shape [B, N_obs].

        Returns:
            Z: Inferred prototype matrix. Shape [B, K, D].
               Z[b, k, :] = prototype k for sample b.
               All K computed independently and simultaneously.
            attn_weights: Attention distributions. Shape [B, K, N_obs].
               attn_weights[b, k, n] = how much prototype k of sample b
               attends to context token n. Used for L_focus loss and
               for interpretability / visualisation.
        """
        B, N_obs, _ = O.shape
        K = P.shape[0]

        # Normalise context (regularisation analog)
        O_norm = self.context_norm(O)  # [B, N_obs, D]

        # Broadcast P across batch dimension
        # P: [K, D] → [B, K, D]  (same queries for every sample)
        Q = P.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Cross-attention: all K prototype slots query O simultaneously.
        # Q = [B, K, D]      (prototype queries — one row per slot)
        # K = V = [B, N_obs, D]  (observed context — image + text tokens)
        # Output: [B, K, D]  (attended representation per slot)
        # attn_weights: [B, K, N_obs] (one probability distribution per slot)
        #
        # Independence: slot k's output depends only on Q[b,k,:] = P[k,:] and O.
        # The MultiheadAttention operation is a matrix multiplication:
        # output[b,k,:] = sum_n attn_weights[b,k,n] * O[b,n,:]
        # No output[b,j,:] appears on the right-hand side for j≠k.
        attended, attn_weights = self.cross_attn(
            query=Q,
            key=O_norm,
            value=O_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # average over heads for interpretability
        )
        # attended:    [B, K, D]
        # attn_weights: [B, K, N_obs]

        # Project to final prototype space
        # proj is applied to each of the K rows independently (shared weights)
        Z = self.proj(attended)  # [B, K, D]

        return Z, attn_weights
```

---

## Step 3: Modify qwen_lvr_model.py

**File:** `src/model/qwen_lvr_model.py`

**In the `QwenWithLVR.__init__` method, add the following after existing head initialisation.**

```python
# ── Import at top of file ──────────────────────────────────────────────────
from src.model.lvr_heads import PrototypeBank, PrototypeCrossAttention
from src.config.prototype_lvr_config import PrototypeLVRConfig

# ── Inside QwenWithLVR.__init__ ───────────────────────────────────────────
# Add after existing LVR head setup:

if hasattr(config, 'prototype_config') and config.prototype_config is not None:
    proto_cfg: PrototypeLVRConfig = config.prototype_config
    D = config.hidden_size  # 2048 for Qwen 3B

    # P: learnable prototype bank — K trainable query vectors.
    # These are NOT produced by any network. They are raw nn.Parameters
    # updated directly by backprop through L_CE + L_div + L_focus.
    self.prototype_bank = PrototypeBank(
        K=proto_cfg.num_prototypes,
        D=D,
        init_scale=proto_cfg.prototype_init_scale,
    )

    # g_φ: prototype cross-attention module.
    # Maps (P, O) → Z where O = concat(V, C).
    # Guarantees ∂z_k/∂z_j = 0 for k≠j by architecture.
    self.prototype_cross_attn = PrototypeCrossAttention(
        D=D,
        num_heads=proto_cfg.prototype_num_heads,
        dropout=proto_cfg.prototype_dropout,
    )

    # Storage for attention weights (used by loss in trainer)
    self._last_proto_attn_weights = None  # [B, K, N_obs]
    self._last_proto_Z = None             # [B, K, D]

    # Token IDs for [proto_1]...[proto_K] in the vocabulary.
    # Set by setup_proto_tokens() after tokenizer is available.
    self.proto_token_ids = None

    print(f"Activated prototype LVR: K={proto_cfg.num_prototypes}, "
          f"D={D}, heads={proto_cfg.prototype_num_heads}")
    print(f"  PrototypeBank params: {proto_cfg.num_prototypes * D:,}")
    print(f"  PrototypeCrossAttn params: "
          f"{sum(p.numel() for p in self.prototype_cross_attn.parameters()):,}")

def setup_proto_tokens(self, tokenizer):
    """
    Add [proto_1]...[proto_K] to the tokenizer vocabulary and register
    their token IDs. Called after model and tokenizer are both loaded.

    These tokens are placeholder tokens in the sequence that get replaced
    with actual prototype embeddings Z in the forward pass — exactly like
    how [lvr] tokens are replaced with visual embeddings in original LVR.
    """
    K = self.prototype_bank.K
    proto_tokens = [f"[proto_{k}]" for k in range(K)]
    tokenizer.add_special_tokens({"additional_special_tokens": proto_tokens})
    self.resize_token_embeddings(len(tokenizer))

    self.proto_token_ids = [
        tokenizer.convert_tokens_to_ids(f"[proto_{k}]") for k in range(K)
    ]
    print(f"Registered proto token IDs: {self.proto_token_ids}")
```

---

## Step 4: Modify monkey_patch_forward_lvr.py

**File:** `src/train/monkey_patch_forward_lvr.py`

**Add the following function and integrate it into the forward pass.**

The original LVR injection (around line 265–301) does:
1. Scatter image embeddings into image placeholder positions
2. Scatter ROI visual embeddings into [lvr] positions sequentially

Replace step 2 with parallel prototype inference. Step 1 (image injection) is unchanged.

```python
def build_observed_context(
    image_embeds: torch.Tensor,    # [N_v, D]  all visual token embeddings
    text_embeds: torch.Tensor,     # [N_c, D]  text token embeddings
) -> torch.Tensor:
    """
    Build the observed context O = concat(V, C).

    O is what the prototype module is allowed to condition on.
    It must NEVER contain any z_k values — prototypes are inferred
    FROM O, not as part of O. This enforces the DIMV principle:
    condition only on truly observed quantities.

    Unlike the shortcut-masking needed for ROI-based imputation,
    here there is no risk of copying because:
    - The prototype slots [proto_k] are randomly initialised tokens
    - Their true "target" is not a fixed position in V — it is whatever
      information helps answer the question (determined by L_CE)
    - There is nothing to copy from O

    Args:
        image_embeds: Visual token embeddings from ViT. Shape [N_v, D].
        text_embeds: LLM token embeddings for the question. Shape [N_c, D].

    Returns:
        O: Observed context. Shape [1, N_v+N_c, D] (batch dim added).
    """
    O = torch.cat([image_embeds, text_embeds], dim=0)  # [N_v+N_c, D]
    return O.unsqueeze(0)                               # [1, N_v+N_c, D]


def inject_prototypes_into_sequence(
    model,
    inputs_embeds: torch.Tensor,  # [B, seq_len, D]
    input_ids: torch.Tensor,       # [B, seq_len]
    image_embeds: torch.Tensor,    # [N_v, D]  (single sample)
    text_embeds: torch.Tensor,     # [N_c, D]  (single sample)
    batch_idx: int,
):
    """
    Run parallel prototype inference and inject all K results into
    inputs_embeds at the [proto_k] token positions.

    Mathematical operation:
        O = concat(V, C)                          [1, N_v+N_c, D]
        Z, A = g_φ(P, O)                          Z:[1,K,D], A:[1,K,N_obs]
        inputs_embeds[b, s_k] ← z_k   ∀k         (all K simultaneously)

    No z_j appears in the computation of z_k for j≠k.
    All K assignments happen in a single tensor operation.

    Args:
        model: QwenWithLVR model instance (has prototype_bank, prototype_cross_attn)
        inputs_embeds: Full input embedding sequence to modify in-place.
        input_ids: Token ID sequence for finding [proto_k] positions.
        image_embeds: Visual embeddings for this sample. Shape [N_v, D].
        text_embeds: Text embeddings for this sample. Shape [N_c, D].
        batch_idx: Which batch item to process (b index).

    Returns:
        inputs_embeds: Modified in-place with prototypes injected.
        attn_weights: Shape [K, N_obs] for loss computation.
    """
    # Build observed context O
    O = build_observed_context(image_embeds, text_embeds)  # [1, N_v+N_c, D]

    # Retrieve learnable prototype bank P
    P = model.prototype_bank()   # [K, D]

    # Parallel inference: all K prototypes in ONE forward pass
    # Z: [1, K, D], attn_weights: [1, K, N_obs]
    Z, attn_weights = model.prototype_cross_attn(P=P, O=O)
    Z = Z.squeeze(0)             # [K, D]
    attn_weights = attn_weights.squeeze(0)  # [K, N_obs]

    # Find [proto_k] positions in the sequence for this batch item
    # proto_token_ids is a list of K token IDs: [id_of_proto_0, ..., id_of_proto_{K-1}]
    proto_positions = []
    for k, tok_id in enumerate(model.proto_token_ids):
        # Find position of [proto_k] in input_ids[batch_idx]
        positions = (input_ids[batch_idx] == tok_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            proto_positions.append((k, positions[0].item()))

    # Inject all K prototype vectors simultaneously — no loop dependency
    # inputs_embeds[batch_idx, s_k] ← z_k   for all k at once
    if proto_positions:
        ks = torch.tensor([k for k, _ in proto_positions], device=Z.device)
        seq_pos = torch.tensor([s for _, s in proto_positions], device=Z.device)
        inputs_embeds[batch_idx, seq_pos] = Z[ks]

    return inputs_embeds, attn_weights


# ── Integration point in the main forward function ────────────────────────
# Find the section that currently handles LVR token injection (around line 277).
# AFTER the existing image embedding injection (masked_scatter for image_mask),
# ADD the following block:

# if model has prototype_bank (prototype mode active):
if hasattr(model, 'prototype_bank') and model.proto_token_ids is not None:
    all_attn_weights = []

    for b in range(batch_size):
        # Get image and text embeddings for this sample
        # image_embeds_b: [N_v, D] — visual tokens for sample b
        # text_embeds_b:  [N_c, D] — text token embeddings for sample b
        # (extract from inputs_embeds using image_mask and text positions)

        inputs_embeds, attn_w = inject_prototypes_into_sequence(
            model=model,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            image_embeds=image_embeds_b,  # [N_v, D]
            text_embeds=text_embeds_b,     # [N_c, D]
            batch_idx=b,
        )
        all_attn_weights.append(attn_w)

    # Store for loss computation in trainer
    model._last_proto_attn_weights = torch.stack(all_attn_weights, dim=0)  # [B, K, N_obs]
    model._last_proto_Z = model.prototype_cross_attn(
        model.prototype_bank(),
        torch.cat([image_embeds, text_embeds], dim=1),
    )[0]  # [B, K, D]
```

---

## Step 5: Modify lvr_trainer.py

**File:** `src/trainer/lvr_trainer.py`

**Add the following loss function and integrate it into the training step.**

```python
import torch
import torch.nn.functional as F


def compute_prototype_losses(
    Z: torch.Tensor,               # [B, K, D]  inferred prototype vectors
    attn_weights: torch.Tensor,    # [B, K, N_obs]  attention distributions
    lambda_div: float = 0.05,
    lambda_focus: float = 0.01,
) -> dict:
    """
    Compute diversity and focus losses for the prototype module.

    These two losses work together with L_CE to produce a good prototype bank:
    - L_CE:    drives prototypes to contain task-useful information
    - L_div:   forces prototypes to capture DIFFERENT information (no collapse)
    - L_focus: forces each prototype to focus on SPECIFIC image regions (compact)

    Without L_div: gradient descent pushes all K slots toward the same vector
    (the mode of the image feature distribution). Z becomes rank-1 despite
    having K rows — all benefit of multiple slots is lost (slot collapse).

    Without L_focus: each slot spreads attention uniformly over all N_obs
    context tokens — diffuse, uninterpretable, not compact.

    Args:
        Z: Inferred prototype matrix from PrototypeCrossAttention.
           Shape [B, K, D]. Each Z[b,k,:] is one prototype for sample b.
        attn_weights: Attention distributions from cross-attention.
           Shape [B, K, N_obs]. Each row sums to 1 (softmax output).
           attn_weights[b,k,n] = how much prototype k of sample b
           attends to context token n.
        lambda_div: Weight for L_div. Increase if slot collapse detected.
        lambda_focus: Weight for L_focus. Decrease if attention over-focuses.

    Returns:
        dict with keys:
            loss_total: λ_1·L_div + λ_2·L_focus (add to L_CE in training step)
            loss_div:   scalar, prototype diversity loss
            loss_focus: scalar, attention entropy loss
            mean_cosine_sim: scalar, diagnostic — mean pairwise cosine similarity
                            (monitor for slot collapse: >0.8 is bad)
    """

    # ── L_div: prototype diversity loss ─────────────────────────────────
    # Objective: make all prototype pairs orthogonal (cosine sim → 0)
    #
    # Formula: L_div = (1 / K(K-1)) · Σ_{j≠k} (z_j · z_k / ‖z_j‖‖z_k‖)²
    #
    # Steps:
    # 1. Normalise each prototype to unit sphere → cosine sim = dot product
    # 2. Compute K×K cosine similarity matrix
    # 3. Square all off-diagonal elements (penalise both + and - correlations)
    # 4. Average over all unique pairs

    # Normalise: each z_k to unit norm
    Z_norm = F.normalize(Z, dim=-1)          # [B, K, D], each row has ‖z_k‖=1

    # Cosine similarity matrix between all pairs of slots
    # sim[b, j, k] = cosine similarity between prototype j and k for sample b
    sim = torch.bmm(Z_norm, Z_norm.transpose(1, 2))  # [B, K, K]

    # Off-diagonal mask: True at positions j≠k
    K = Z.shape[1]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=Z.device)  # [K, K]

    # Extract off-diagonal similarities and square them
    # sim[:, off_diag]: [B, K*(K-1)] — all unique pair similarities
    loss_div = (sim[:, off_diag] ** 2).mean()
    # When loss_div → 0: all pairs orthogonal (good diversity)
    # When loss_div → 1: all slots identical (slot collapse, bad)

    # Diagnostic: mean absolute cosine similarity (easier to interpret)
    mean_cosine_sim = sim[:, off_diag].abs().mean().item()

    # ── L_focus: attention entropy loss ─────────────────────────────────
    # Objective: each slot should focus on a small set of context tokens
    # (low entropy = focused, compact, interpretable)
    # (high entropy = uniform over all tokens = diffuse, bad)
    #
    # Formula: L_focus = (1/K) · Σ_k H(A_k)
    #          where H(A_k) = -Σ_n A_{k,n} · log(A_{k,n})
    #
    # Note: attn_weights is already softmax-normalised (each row sums to 1)
    # Maximum possible entropy = log(N_obs) ≈ log(912) ≈ 6.8 nats
    # Good range after training: entropy per slot ≈ 2–4 nats

    eps = 1e-9  # numerical stability for log
    # Entropy for each slot: -sum_n A_{k,n} * log(A_{k,n})
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    # entropy shape: [B, K]
    loss_focus = entropy.mean()  # average over batch and slots

    # ── Total auxiliary loss ─────────────────────────────────────────────
    loss_total = lambda_div * loss_div + lambda_focus * loss_focus

    return {
        "loss_total": loss_total,
        "loss_div": loss_div.item(),
        "loss_focus": loss_focus.item(),
        "mean_cosine_sim": mean_cosine_sim,
        "mean_entropy": entropy.mean().item(),
    }


# ── Integration in the training step ─────────────────────────────────────
# In compute_loss() or the training_step(), REPLACE:
#
#   loss = loss_CE + loss_lvr_lambda * loss_LVR
#
# WITH:
#
#   proto_losses = compute_prototype_losses(
#       Z=model._last_proto_Z,
#       attn_weights=model._last_proto_attn_weights,
#       lambda_div=proto_cfg.loss_diversity_lambda,
#       lambda_focus=proto_cfg.loss_focus_lambda,
#   )
#   loss = loss_CE + proto_losses["loss_total"]
#
# AND LOG:
#   self.log("loss_CE",           loss_CE,                          ...)
#   self.log("loss_div",          proto_losses["loss_div"],          ...)
#   self.log("loss_focus",        proto_losses["loss_focus"],        ...)
#   self.log("mean_cosine_sim",   proto_losses["mean_cosine_sim"],   ...)
#   self.log("mean_entropy",      proto_losses["mean_entropy"],      ...)
#
# SLOT COLLAPSE DIAGNOSTIC (check every 100 steps):
#   if proto_losses["mean_cosine_sim"] > 0.8:
#       logger.warning("Slot collapse detected! "
#                      f"mean_cosine_sim={proto_losses['mean_cosine_sim']:.3f}. "
#                      "Increase lambda_div or reduce LR for prototype_bank.")
```

---

## Step 6: Modify Dataset (lvr_sft_dataset_packed.py)

**File:** `src/dataset/lvr_sft_dataset_packed.py`

**In the conversation building function that currently inserts `<lvr>` placeholders,
add a parallel path for prototype tokens.**

```python
def build_prototype_token_sequence(
    conversation: list,
    K: int,
    proto_token_template: str = "[proto_{k}]",
) -> str:
    """
    Insert K prototype slot tokens into the conversation at the position
    where visual reasoning should occur.

    The prototype tokens replace the <lvr> placeholder used in original LVR.
    Instead of:
        "Look at this region: <lvr>. What is the object?"
        → expanded to: [lvr_start][lvr]×K[lvr_end]

    We use:
        "Analyse the image: <proto>. What is the object?"
        → expanded to: [proto_0][proto_1]...[proto_{K-1}]

    Note: There is NO [proto_start] or [proto_end] wrapper.
    The K prototype tokens are inserted directly. The LLM learns to
    read them as a sequence of K visual summary vectors.

    Args:
        conversation: List of {"role": ..., "content": ...} dicts.
        K: Number of prototype slots (from config.num_prototypes).
        proto_token_template: Template string for each token.

    Returns:
        Modified conversation text with prototype tokens inserted.
    """
    proto_span = " ".join(
        proto_token_template.format(k=k) for k in range(K)
    )
    # Replace <proto> placeholder with the K prototype tokens
    # The dataset should use <proto> as the placeholder in training data
    for turn in conversation:
        if "<proto>" in turn.get("content", ""):
            turn["content"] = turn["content"].replace("<proto>", proto_span)
    return conversation
```

---

## Step 7: Training Script Changes

**File:** `train-stage1-3b-prototype.sh` (new script, based on existing train-stage1-3b-lora.sh)

**Key changes from the original training script:**

```bash
# Add to training arguments:
--prototype_mode True \
--num_prototypes 8 \
--prototype_num_heads 8 \
--loss_diversity_lambda 0.05 \
--loss_focus_lambda 0.01 \
--warmup_steps_prototype_only 500 \

# Frozen components (same as original):
--freeze_vision_tower True \
--freeze_merger True \

# LoRA settings (same as original):
--lora_rank 64 \
--lora_alpha 128 \

# Learning rates — prototype module needs higher LR than LoRA:
--learning_rate 2e-4 \          # for LoRA adapters
--prototype_lr 5e-4 \           # for P and g_phi (starts from random)
```

---

## Step 8: Parameter Groups for Optimiser

**In the trainer, set up separate parameter groups** so P and g_φ can have a
different learning rate from the LoRA adapters.

```python
def get_parameter_groups(model, proto_lr: float = 5e-4, lora_lr: float = 2e-4):
    """
    Separate parameter groups for prototype module vs LoRA adapters.

    Prototype module (P and g_φ) needs higher LR because:
    - P is randomly initialised (nn.Parameter from scratch)
    - g_φ weights are randomly initialised
    - LoRA adapters are near-identity initialised (B=0 at start)

    Args:
        model: QwenWithLVR with prototype modules.
        proto_lr: LR for PrototypeBank and PrototypeCrossAttention.
        lora_lr: LR for LoRA adapters.

    Returns:
        List of parameter group dicts for the optimiser.
    """
    prototype_params = []
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "prototype_bank" in name or "prototype_cross_attn" in name:
            prototype_params.append(param)
        elif "lora" in name.lower():
            lora_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": prototype_params, "lr": proto_lr,  "name": "prototype"},
        {"params": lora_params,      "lr": lora_lr,   "name": "lora"},
        {"params": other_params,     "lr": lora_lr,   "name": "other"},
    ]
```

---

## Monitoring and Diagnostics

### Metrics to Log Every Step
| Metric | Expected range | Warning if |
|--------|---------------|-----------|
| `loss_CE` | Decreases steadily | Diverges after step 500 |
| `loss_div` | 0.8→0.1 over training | Stays >0.8 after 1000 steps |
| `loss_focus` | log(N_obs)→2–4 | Stays >6 (uniform attention) |
| `mean_cosine_sim` | 1.0→0.1–0.3 | Stays >0.8 (slot collapse) |
| `mean_entropy` | log(N_obs)→2–4 | Same as loss_focus |

### Slot Collapse: Detection and Fix
```
Detection: mean_cosine_sim > 0.8 after 500 steps
Fix 1: Increase loss_diversity_lambda (0.05 → 0.1 → 0.2)
Fix 2: Reduce prototype_lr (5e-4 → 1e-4) — slower but more stable
Fix 3: Orthogonal initialisation for P instead of random normal
```

### Verifying Independence (unit test to add)
```python
def test_prototype_independence(model, sample_O):
    """
    Verify that changing z_k does not affect z_j for j≠k.
    This should ALWAYS pass — independence is guaranteed by architecture.
    If this test fails, the implementation has a bug.
    """
    P = model.prototype_bank()  # [K, D]
    Z1, _ = model.prototype_cross_attn(P, sample_O)

    # Perturb only p_0 (first prototype query)
    P_perturbed = P.clone()
    P_perturbed[0] += torch.randn_like(P[0])
    Z2, _ = model.prototype_cross_attn(P_perturbed, sample_O)

    # z_0 should change (we perturbed p_0)
    assert not torch.allclose(Z1[0, 0], Z2[0, 0], atol=1e-6), \
        "z_0 did not change after perturbing p_0"

    # z_1...z_{K-1} should be EXACTLY unchanged (independence)
    for k in range(1, P.shape[0]):
        assert torch.allclose(Z1[0, k], Z2[0, k], atol=1e-6), \
            f"z_{k} changed after perturbing p_0 — independence violated!"

    print("Independence test passed: ∂z_k/∂z_j = 0 for all k≠j ✓")
```

---

## Summary: What Changes vs What Stays

| Component | Original LVR | Prototype LVR |
|-----------|-------------|---------------|
| Token type | `[lvr_start][lvr]×K[lvr_end]` | `[proto_0]...[proto_{K-1}]` |
| Token source | ROI bounding box indices → V_ROI | Learnable P + cross-attention → Z |
| Generation | Sequential: z_k depends on z_{k-1} | Parallel: all z_k from O simultaneously |
| Error growth | ≥ K·σ² (grows with K) | σ² (constant) |
| Region selection | Hard integer indices (not differentiable) | Soft attention over all patches (differentiable) |
| Loss | L_CE + λ·L_MSE(h^lvr_start, v*) | L_CE + λ_1·L_div + λ_2·L_focus |
| Annotation needed | Yes (bounding boxes required) | No (annotation-free) |
| New trainable params | None beyond LoRA | P [K×D] + g_φ [~33M] |
| Gradient to region | Zero (hard index) | Non-zero (through attention weights A) |