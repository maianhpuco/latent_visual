"""
DIMV-style Latent Reasoning Module.

Computes Z^(final) ∈ R^{T_v × d} from observed context X_o ∈ R^{N_obs × d}
via L iterations of cross-attention refinement.

Mathematical correspondence with DIMV:
    DIMV:  Ê[X_miss | X_obs] = Σ_mo (Σ_oo + αI)^{-1} X_obs   (linear, Gaussian)
    Here:  Z^(final)         = Attn(Q=Z, K=X_o, V=X_o)        (nonlinear, per-image)

The cross-attention weight matrix A ∈ R^{T_v × N_obs} plays the role of
Σ_mo (Σ_oo + αI)^{-1}: it learns which context tokens matter for each slot.

Key property — parallel inference:
    All T_v slots are updated simultaneously at each step.
    ∂z_k / ∂z_j = 0 for k ≠ j throughout this module.
    (Slots interact only through the LLM's self-attention within the Z block
     of the sequence — not through this module.)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIPooler(nn.Module):
    """
    Pools K ROI visual token embeddings → T_v imputation targets.
    Maps V_ROI* ∈ R^{K×d} → Ẑ_ROI ∈ R^{T_v×d}.
    Used only for constructing L_IMP targets; not on the inference path.
    """

    def __init__(self, d: int, T_v: int, method: str = "attention_pool", num_heads: int = 8):
        super().__init__()
        self.d = d
        self.T_v = T_v
        self.method = method

        if method == "attention_pool":
            self.pool_queries = nn.Parameter(torch.randn(T_v, d) * (d ** -0.5))
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=num_heads, dropout=0.0, batch_first=True
            )

    def forward(
        self,
        V_roi: torch.Tensor,                    # [B, K, d]
        roi_padding_mask: torch.Tensor = None,  # [B, K] True=padding
    ) -> torch.Tensor:
        B = V_roi.shape[0]

        if self.method == "attention_pool":
            Q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.pool_attn(
                query=Q, key=V_roi, value=V_roi, key_padding_mask=roi_padding_mask
            )
            return pooled  # [B, T_v, d]

        elif self.method == "adaptive_pool":
            x = V_roi.transpose(1, 2)  # [B, d, K]
            if roi_padding_mask is not None:
                x = x * (1.0 - roi_padding_mask.unsqueeze(1).float())
            return F.adaptive_avg_pool1d(x, self.T_v).transpose(1, 2)  # [B, T_v, d]

        elif self.method == "avg_pool":
            if roi_padding_mask is not None:
                valid = (~roi_padding_mask).float().unsqueeze(-1)  # [B, K, 1]
                n_valid = valid.sum(dim=1).clamp(min=1.0)
                global_avg = (V_roi * valid).sum(dim=1) / n_valid  # [B, d]
            else:
                global_avg = V_roi.mean(dim=1)
            return global_avg.unsqueeze(1).expand(-1, self.T_v, -1)

        else:
            raise ValueError(f"Unknown roi_pool_method: {self.method}")


def compute_imputation_loss(
    Z_final: torch.Tensor,       # [B, T_v, d]
    Z_roi_target: torch.Tensor,  # [B, T_v, d]
    loss_type: str = "cosine",
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute L_IMP = d(Z^(final), Ẑ_ROI). Supports cosine, mse, nce."""
    B, T_v, d = Z_final.shape
    Z_f = Z_final.float()
    Z_t = Z_roi_target.float()

    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(Z_f, Z_t, dim=-1)).mean()

    elif loss_type == "mse":
        return F.mse_loss(Z_f, Z_t, reduction="mean")

    elif loss_type == "nce":
        Z_f_flat = F.normalize(Z_f.view(B * T_v, d), dim=-1)
        Z_t_flat = F.normalize(Z_t.view(B * T_v, d), dim=-1)
        logits = Z_f_flat @ Z_t_flat.T / temperature
        labels = torch.arange(B * T_v, device=Z_final.device)
        return F.cross_entropy(logits, labels)

    else:
        raise ValueError(f"Unknown imputation_loss_type: {loss_type}")


class LatentReasoningModule(nn.Module):
    """
    Iterative cross-attention module that imputes Z from observed context X_o.

    Args:
        d           LLM hidden size (2048 for Qwen 3B).
        T_v         Number of latent reasoning slots.
        L           Number of refinement iterations (default 2).
        num_heads   Attention heads. Must divide d.
        dropout     Attention dropout (0.0 recommended for fine-tuning).
        use_layer_norm  Apply LN in residual update (strongly recommended).
        use_ffn     Apply 4x FFN after cross-attention.
        slot_init   One of "learned" | "zero" | "gaussian" | "last_hidden".
    """

    def __init__(
        self,
        d: int,
        T_v: int,
        L: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_ffn: bool = True,
        slot_init: str = "learned",
    ):
        super().__init__()
        self.d = d
        self.T_v = T_v
        self.L = L
        self.slot_init = slot_init
        self.use_layer_norm = use_layer_norm
        self.use_ffn = use_ffn

        # ── Learnable slot queries Z^(0) ─────────────────────────────────
        # Shape [T_v, d]. Each slot has a distinct learned initialisation.
        # Initialised with small normal noise (scale = 1/sqrt(d)) to break
        # symmetry while staying in the same scale as the LLM embeddings.
        if slot_init == "learned":
            self.slot_queries = nn.Parameter(
                torch.randn(T_v, d) * (d ** -0.5)
            )
        else:
            self.slot_queries = None

        # ── Cross-attention (shared across all L refinement steps) ───────
        # Q comes from Z^(t), K and V come from X_o.
        # batch_first=True: [B, seq, d] layout.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── LayerNorm ────────────────────────────────────────────────────
        if use_layer_norm:
            self.norm_q  = nn.LayerNorm(d)   # pre-norm on slot queries
            self.norm_out = nn.LayerNorm(d)  # post-norm after residual update

        # ── Feed-forward network (4x expansion) ──────────────────────────
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Linear(d * 4, d),
            )

    def _init_slots(
        self,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        last_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct Z^(0) of shape [B, T_v, d].

        "learned"     — expand the shared learnable parameter across the batch.
        "zero"        — all zeros.
        "gaussian"    — N(0, 0.02·I) per forward call (stochastic).
        "last_hidden" — repeat of the last observed token's hidden state.
        """
        if self.slot_init == "learned":
            # [T_v, d] → [B, T_v, d]
            return self.slot_queries.to(dtype=dtype).unsqueeze(0).expand(B, -1, -1)

        elif self.slot_init == "zero":
            return torch.zeros(B, self.T_v, self.d, device=device, dtype=dtype)

        elif self.slot_init == "gaussian":
            return torch.randn(B, self.T_v, self.d, device=device, dtype=dtype) * 0.02

        elif self.slot_init == "last_hidden":
            assert last_hidden is not None, (
                "last_hidden must be provided when slot_init='last_hidden'"
            )
            # last_hidden: [B, d] → [B, T_v, d]
            return last_hidden.unsqueeze(1).expand(-1, self.T_v, -1)

        else:
            raise ValueError(f"Unknown slot_init: {self.slot_init!r}")

    def forward(
        self,
        X_o: torch.Tensor,                              # [B, N_obs, d]
        last_hidden: Optional[torch.Tensor] = None,    # [B, d] for last_hidden init
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N_obs], True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run L cross-attention refinement steps to produce Z^(final).

        Args:
            X_o:              Observed context. Shape [B, N_obs, d].
                              = concat(V_masked_out, x_txt) in DIMV terminology.
                              Must NOT contain any Z slot tokens (they are missing).
            last_hidden:      Optional [B, d] for slot_init="last_hidden".
            key_padding_mask: Optional [B, N_obs]. True marks padding positions
                              in X_o that should be ignored by attention.

        Returns:
            Z_final:     [B, T_v, d] — final imputed reasoning tokens.
            attn_weights:[B, T_v, N_obs] — attention weights from the final step
                         (useful for visualisation; attn_weights[b,k,n] = how much
                          slot k of sample b attended to context token n).
        """
        B = X_o.shape[0]
        device = X_o.device
        dtype = X_o.dtype

        # Z^(0): initial slot embeddings [B, T_v, d]
        Z = self._init_slots(B, device, dtype, last_hidden)

        attn_weights = None

        for _ in range(self.L):
            # Pre-norm on queries (stabilises training)
            Q = self.norm_q(Z) if self.use_layer_norm else Z  # [B, T_v, d]

            # Cross-attention: Q from slots, K/V from observed context
            # All T_v slots attend to X_o simultaneously — fully parallel.
            # No slot z_k attends to any other slot z_j here.
            Z_attended, attn_weights = self.cross_attn(
                query=Q,
                key=X_o,
                value=X_o,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,   # [B, T_v, N_obs]
            )
            # Z_attended: [B, T_v, d]

            # Optional feed-forward nonlinearity
            Z_update = self.ffn(Z_attended) if self.use_ffn else Z_attended

            # Residual update with optional post-norm
            if self.use_layer_norm:
                Z = self.norm_out(Z + Z_update)
            else:
                Z = Z + Z_update

        return Z, attn_weights   # Z = Z^(final) [B, T_v, d]
