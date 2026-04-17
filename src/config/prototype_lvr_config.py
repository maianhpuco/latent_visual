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
