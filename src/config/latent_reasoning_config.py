from dataclasses import dataclass


@dataclass
class LatentReasoningConfig:
    """
    Configuration for DIMV-style latent reasoning token imputation.

    Core principle:
        Sequence layout: [x_txt | V | Z | Y]
        Z = T_v latent slots computed in parallel from X_o = concat(V, x_txt).
        Attention mask blocks Y from attending to V directly.
        NTP loss alone is sufficient — gradients for Y flow exclusively through Z.

    DIMV analogy:
        X_o (observed) = image patches V + text prefix x_txt
        Z   (missing)  = latent reasoning tokens, imputed via cross-attention
        The cross-attention Attn(Q=Z, K=X_o, V=X_o) is the learned analog of
        DIMV's linear imputation Σ_mo (Σ_oo + αI)^{-1} X_o.
    """

    # ── Slot parameters ────────────────────────────────────────────────────
    num_reasoning_slots: int = 64
    """
    T_v = number of latent reasoning tokens (the bottleneck width).
    Small T_v (16-32): tight bottleneck, heavy compression.
    Large T_v (64-128): looser bottleneck, easier to encode detail.
    Default 64. Must be << N (≈784 image tokens) to ensure compression.
    """

    slot_init: str = "learned"
    """
    How to initialise Z^(0) before iterative cross-attention refinement.
    Options:
      "learned"     — nn.Parameter [T_v, d]. Recommended default.
      "zero"        — all zeros; diversity emerges from cross-attention only.
      "gaussian"    — N(0, 0.02·I) per forward pass; stochastic.
      "last_hidden" — repeat of the last observed token's hidden state.
    """

    num_refinement_steps: int = 2
    """
    L = number of iterative cross-attention refinement steps.
    Each step: Z^(t+1) = LN(Z^(t) + FFN(Attn(Q=Z^(t), K=X_o, V=X_o)))
    L=1: fast, often sufficient.
    L=2: allows one correction after the initial read. Recommended.
    """

    num_attn_heads: int = 8
    """
    Attention heads in the cross-attention module.
    Must divide hidden_size evenly: 2048 / 8 = 256 per head for Qwen 3B.
    """

    dropout: float = 0.0
    """Dropout in cross-attention. Set 0.0 for fine-tuning stability."""

    use_layer_norm: bool = True
    """
    Apply LayerNorm in the residual update:
        Z^(t+1) = LN(Z^(t) + FFN(Attn(...)))
    Strongly recommended. Without it, slot magnitudes grow across iterations.
    """

    use_ffn: bool = True
    """
    Apply a 4x feed-forward network after cross-attention before residual update.
    Adds nonlinear transformation capacity.
    """

    # ── ROI imputation supervision ─────────────────────────────────────────
    use_roi_supervision: bool = False
    """
    If True: add L_IMP loss — Z^(final) must match pooled ROI visual embeddings.
    If False: fall back to NTP-only (original DIMV baseline behaviour).
    """

    roi_pool_method: str = "attention_pool"
    """
    How to reduce K ROI visual tokens to T_v imputation targets.
    Options: "attention_pool" | "adaptive_pool" | "avg_pool"
    """

    imputation_loss_type: str = "cosine"
    """
    Distance metric for L_IMP. Options: "cosine" | "mse" | "nce"
    """

    imputation_loss_lambda: float = 0.1
    """λ weight for L_IMP. L = L_NTP + λ * L_IMP. Safe range: 0.05–0.5."""

    nce_temperature: float = 0.07
    """Temperature τ for NCE imputation loss. Only used when imputation_loss_type="nce"."""

    # ── Checkpoint and validation ──────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_dimv_roi"
    """Directory for all DIMV-ROI checkpoints. Never touches existing checkpoints/."""

    vstar_val_fraction: float = 0.30
    """Fraction of v*star dataset to use for validation (sampled ONCE at training start)."""

    validate_every_n_steps: int = 500
    """Run v*star validation after every N training steps."""

    early_checkpoint_steps: list = None
    """Extra early checkpoint steps, e.g. [10, 50, 100]. Created ONCE, never deleted."""

    vstar_val_seed: int = 42
    """Fixed seed for sampling the 30% validation set."""
