import torch
import torch.nn as nn
from torch.nn import LayerNorm

class LVRHead(nn.Module):
    """
        The simplest mlp w/o up_proj
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


from transformers.activations import ACT2FN
class LVRHeadGLU(nn.Module):
    ''' 
        The Gated Liner Unit MLP
    '''
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        # 11008 for 3b; 18944 for 7b; 27648 for 32b
        self.intermediate_size = intermediate_size  
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[self.hidden_act]    #silu

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    



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
        P: torch.Tensor,                       # [K, D] prototype bank
        O: torch.Tensor,                       # [B, N_obs, D] observed context
        key_padding_mask: torch.Tensor = None, # [B, N_obs] True=padding position
    ):
        """
        Infer all K prototype vectors in one parallel forward pass.

        Args:
            P: Prototype bank from PrototypeBank.forward(). Shape [K, D].
               Same for all batch items — will be broadcast across B.
            O: Observed context = concat(V_masked, C). Shape [B, N_obs, D].
               IMPORTANT: O must NOT contain any z_k values.
            key_padding_mask: True at padding positions. Shape [B, N_obs].

        Returns:
            Z: Inferred prototype matrix. Shape [B, K, D].
            attn_weights: Attention distributions. Shape [B, K, N_obs].
        """
        B, N_obs, _ = O.shape

        # Normalise context (regularisation analog)
        O_norm = self.context_norm(O)  # [B, N_obs, D]

        # Broadcast P across batch dimension: [K, D] → [B, K, D]
        Q = P.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # Cross-attention: all K prototype slots query O simultaneously.
        # Independence: slot k's output depends only on Q[b,k,:] = P[k,:] and O.
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
        Z = self.proj(attended)  # [B, K, D]

        return Z, attn_weights


# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
# class LVRHeadRMS(nn.Module):
#     """
#         Modified Patch Merger from transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
#         This inherits from the mm projector of qwen 2.5 vl
#     """
#     def __init__(self, hidden_size: int) -> None:
#         super().__init__()
#         self.ln_q = Qwen2RMSNorm(hidden_size, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, hidden_size),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp(self.ln_q(x))
#         return x