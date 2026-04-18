import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.utils import is_torchdynamo_compiling,TransformersKwargs
from transformers.processing_utils import Unpack
from src.constants import IGNORE_INDEX
import torch.distributed as dist

import torch.nn.functional as F


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def _extract_image_embeds(features):
    """
    Newer transformers returns BaseModelOutputWithPooling from
    get_image_features(); the actual split tensors live in .pooler_output.
    Older versions return a plain tuple.  Handle both transparently.
    """
    if isinstance(features, (list, tuple)):
        return features
    if hasattr(features, 'pooler_output'):
        return features.pooler_output
    if hasattr(features, 'last_hidden_state'):
        return features.last_hidden_state
    return features


def _build_mm_token_type_ids(input_ids, config):
    """
    Build mm_token_type_ids: (batch_size, seq_len) tensor.
    0 = text, 1 = image placeholder, 2 = video placeholder.
    Required by newer transformers get_rope_index().
    """
    mm_ids = torch.zeros_like(input_ids, dtype=torch.int)
    image_token_id = getattr(config, 'image_token_id', None)
    video_token_id = getattr(config, 'video_token_id', None)
    if image_token_id is not None:
        mm_ids[input_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_ids[input_ids == video_token_id] = 2
    return mm_ids


def _get_rope_index_compat(model, input_ids, image_grid_thw, video_grid_thw,
                           second_per_grid_ts, attention_mask, config):
    """
    Call get_rope_index with compatibility for both old (no mm_token_type_ids)
    and new (requires mm_token_type_ids) transformers versions.
    """
    mm_token_type_ids = _build_mm_token_type_ids(input_ids, config)
    try:
        return model.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )
    except TypeError:
        return model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )


def build_observed_context(
    image_embeds: torch.Tensor,   # [N_v, D]
    text_embeds: torch.Tensor,    # [N_c, D]
) -> torch.Tensor:
    """
    Build O = concat(V, C) as [1, N_v+N_c, D].

    O is what the prototype cross-attention is conditioned on.
    It must NOT contain any z_k values — prototypes are inferred FROM O.
    """
    O = torch.cat([image_embeds, text_embeds], dim=0)  # [N_v+N_c, D]
    return O.unsqueeze(0)                               # [1, N_v+N_c, D]


def inject_prototypes_into_sequence(
    model,
    inputs_embeds: torch.Tensor,   # [B, seq_len, D]
    input_ids: torch.Tensor,        # [B, seq_len]
    image_embeds_b: torch.Tensor,   # [N_v, D] for batch item b
    text_embeds_b: torch.Tensor,    # [N_c, D] for batch item b
    batch_idx: int,
):
    """
    Run parallel prototype inference and inject all K results into
    inputs_embeds at the [proto_k] token positions.

    Returns:
        inputs_embeds: modified in-place with Z injected.
        attn_weights:  [K, N_obs] for loss computation.
        Z:             [K, D] prototype vectors.
    """
    O = build_observed_context(image_embeds_b, text_embeds_b)  # [1, N_v+N_c, D]
    P = model.prototype_bank()                                  # [K, D]

    # Parallel inference: all K prototypes in ONE forward pass
    Z, attn_weights = model.prototype_cross_attn(P=P, O=O)
    Z = Z.squeeze(0)            # [K, D]
    attn_weights = attn_weights.squeeze(0)  # [K, N_obs]

    # Find [proto_k] positions in the sequence for this batch item
    proto_positions = []
    for k, tok_id in enumerate(model.proto_token_ids):
        positions = (input_ids[batch_idx] == tok_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            proto_positions.append((k, positions[0].item()))

    # Inject all K prototype vectors simultaneously
    if proto_positions:
        ks = torch.tensor([k for k, _ in proto_positions], device=Z.device)
        seq_pos = torch.tensor([s for _, s in proto_positions], device=Z.device)
        inputs_embeds[batch_idx, seq_pos] = Z[ks].to(inputs_embeds.dtype)

    return inputs_embeds, attn_weights, Z


def qwen2_5_mixed_modality_forward_prototype(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    # Original LVR fields (unused in prototype mode, kept for interface compat)
    lvr_tokens: Optional[torch.Tensor] = None,
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, "Qwen2_5_VLCausalLMOutputWithPast"]:
    """
    Prototype-based parallel LVR forward pass.

    Replaces sequential [lvr] token injection with K independent prototype
    slots computed via cross-attention: z_k = g_φ(p_k, O), ∂z_k/∂z_j = 0.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # Reset per-step prototype state so stale values from a previous batch are never used
    self._last_proto_Z = None
    self._last_proto_attn_weights = None

    # Handle text-only batches (no pixel values) — dummy visual pass for DeepSpeed ZeRO-3
    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        dummy_image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + dummy_image_embeds.mean() * 0

    batch_size = inputs_embeds.shape[0]
    image_embeds_all = None  # will be set if pixel_values present

    if pixel_values is not None:
        image_embeds_all = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds_all = torch.cat(_extract_image_embeds(image_embeds_all), dim=0)

        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds_all.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_mask = input_ids == self.config.image_token_id  # [B, seq_len]
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds_all = image_embeds_all.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds_all)

        # Per-sample: build observed context O and inject prototype vectors Z
        all_attn_weights = []
        all_Z = []
        total_tokens = torch.sum(image_mask, dim=1)  # [B]: #image tokens per sample
        image_token_offsets = torch.cumsum(
            F.pad(total_tokens, (1, 0)), dim=0
        )[:-1]  # [B]: global offset into image_embeds_all for each sample

        # Build a mask for proto token positions (any [proto_k])
        proto_any_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.proto_token_ids is not None:
            for tok_id in self.proto_token_ids:
                proto_any_mask |= (input_ids == tok_id)

        for b in range(batch_size):
            # Extract visual embeddings for sample b
            offset = image_token_offsets[b].item()
            n_vis = total_tokens[b].item()
            image_embeds_b = image_embeds_all[offset: offset + n_vis]  # [N_v, D]

            # Extract text embeddings: non-image, non-proto positions
            text_mask_b = ~image_mask[b] & ~proto_any_mask[b]  # [seq_len]
            text_embeds_b = inputs_embeds[b][text_mask_b]       # [N_c, D]

            if self.proto_token_ids is not None and len(self.proto_token_ids) > 0:
                inputs_embeds, attn_w, Z = inject_prototypes_into_sequence(
                    model=self,
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    image_embeds_b=image_embeds_b,
                    text_embeds_b=text_embeds_b,
                    batch_idx=b,
                )
                all_attn_weights.append(attn_w)
                all_Z.append(Z)

        if all_attn_weights:
            self._last_proto_attn_weights = torch.stack(all_attn_weights, dim=0)  # [B, K, N_obs]
            self._last_proto_Z = torch.stack(all_Z, dim=0)                        # [B, K, D]

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size_inner, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size_inner, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size_inner, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size_inner // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]
    logits = self.lm_head(hidden_states)

    loss_ce = None
    loss_proto = None

    if labels is not None:
        logits_f = logits.float()
        shift_logits = logits_f[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Mask out [proto_k] token positions from CE loss
        if self.proto_token_ids is not None:
            for tok_id in self.proto_token_ids:
                shift_labels = shift_labels.masked_fill(shift_labels == tok_id, IGNORE_INDEX)

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # Prototype auxiliary losses (diversity + focus)
        if self._last_proto_Z is not None and self._last_proto_attn_weights is not None:
            proto_cfg = getattr(self.config, 'prototype_config', None)
            lambda_div   = proto_cfg.loss_diversity_lambda if proto_cfg else 0.05
            lambda_focus = proto_cfg.loss_focus_lambda     if proto_cfg else 0.01

            Z_f = self._last_proto_Z.float()
            A_f = self._last_proto_attn_weights.float()

            # L_div: mean squared cosine similarity between all prototype pairs
            Z_norm = F.normalize(Z_f, dim=-1)         # [B, K, D]
            sim = torch.bmm(Z_norm, Z_norm.transpose(1, 2))  # [B, K, K]
            K = Z_f.shape[1]
            off_diag = ~torch.eye(K, dtype=torch.bool, device=Z_f.device)
            loss_div = (sim[:, off_diag] ** 2).mean()

            # L_focus: mean attention entropy
            eps = 1e-9
            entropy = -(A_f * (A_f + eps).log()).sum(dim=-1)  # [B, K]
            loss_focus = entropy.mean()

            loss_proto = lambda_div * loss_div + lambda_focus * loss_focus

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss_ce,) + output if loss_ce is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_lvr=loss_proto,          # reuse loss_lvr field for proto aux loss
        loss_mode_switch=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DIMV-style latent reasoning forward pass
# ═══════════════════════════════════════════════════════════════════════════

def _build_dimv_4d_mask(
    input_ids: torch.Tensor,    # [B, seq_len]
    image_token_id: int,
    slot_token_ids: list,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a [B, 1, seq_len, seq_len] additive float attention mask that:
      - Enforces causal order for all positions.
      - Blocks Y (answer tokens, after the last slot token) from attending
        to V (image patch tokens) — this is the bottleneck.
      - Z (slot tokens) can attend to everything before them (causal).

    Values: 0.0 = allowed, float('-inf') = blocked.
    The mask is passed directly to Qwen2_5_VLTextModel; transformers 4.57
    returns any 4-D mask as-is from _preprocess_mask_arguments.
    """
    B, seq_len = input_ids.shape

    # Start with a standard causal mask: upper-triangle = -inf
    causal = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )  # [seq_len, seq_len]

    # Expand to [B, 1, seq_len, seq_len] (broadcast over num_heads)
    mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, seq_len, seq_len).clone()

    for b in range(B):
        # V positions: image placeholder tokens
        v_pos = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        if len(v_pos) == 0:
            continue

        # Z positions: all slot tokens
        slot_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for sid in slot_token_ids:
            slot_mask |= (input_ids[b] == sid)
        z_pos = slot_mask.nonzero(as_tuple=True)[0]
        if len(z_pos) == 0:
            continue

        # Y positions: everything AFTER the last slot token
        last_z = z_pos[-1].item()
        if last_z + 1 >= seq_len:
            continue
        y_pos = torch.arange(last_z + 1, seq_len, device=device)

        # Block Y → V: answer tokens cannot attend to image patches
        mask[b, 0][y_pos.unsqueeze(1), v_pos.unsqueeze(0)] = float('-inf')

    return mask  # [B, 1, seq_len, seq_len]


def _extract_observed_context(
    inputs_embeds: torch.Tensor,   # [B, seq_len, d]
    input_ids: torch.Tensor,        # [B, seq_len]
    slot_token_ids: list,
) -> tuple:
    """
    Extract X_o = all embeddings except slot positions.

    X_o is the "observed" set in DIMV terminology: it contains V (image patches)
    and x_txt (text tokens), but NOT the Z slots (those are the missing variables).

    Returns:
        X_o_batch:    [B, N_obs_max, d], zero-padded to the longest sample.
        key_pad_mask: [B, N_obs_max] bool, True at padding positions.
    """
    B, seq_len, d = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    all_X_o = []
    for b in range(B):
        slot_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for sid in slot_token_ids:
            slot_mask |= (input_ids[b] == sid)
        obs_mask = ~slot_mask  # everything that is NOT a slot token
        all_X_o.append(inputs_embeds[b][obs_mask])  # [N_obs_b, d]

    N_max = max(x.shape[0] for x in all_X_o)
    X_o_batch = torch.zeros(B, N_max, d, device=device, dtype=dtype)
    key_pad_mask = torch.ones(B, N_max, dtype=torch.bool, device=device)

    for b, x in enumerate(all_X_o):
        n = x.shape[0]
        X_o_batch[b, :n] = x
        key_pad_mask[b, :n] = False  # False = not padding

    return X_o_batch, key_pad_mask


def _inject_z_into_embeds(
    inputs_embeds: torch.Tensor,   # [B, seq_len, d] — modified in-place
    input_ids: torch.Tensor,        # [B, seq_len]
    Z_final: torch.Tensor,          # [B, T_v, d]
    slot_token_ids: list,           # list of T_v token IDs in order
) -> torch.Tensor:
    """
    Replace [SLOT_k] placeholder embeddings with Z_final[b, k] for all b, k.
    Operates in-place on inputs_embeds.
    """
    B = Z_final.shape[0]
    for b in range(B):
        for k, sid in enumerate(slot_token_ids):
            positions = (input_ids[b] == sid).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                inputs_embeds[b, positions[0].item()] = Z_final[b, k].to(inputs_embeds.dtype)
    return inputs_embeds


def qwen2_5_mixed_modality_forward_dimv(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    # Original LVR fields — unused in DIMV mode, kept for interface compat
    lvr_tokens: Optional[torch.Tensor] = None,
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    # Absorb extra kwargs from the standard transformers generate loop
    # (e.g. cu_seq_lens_q passed by flash-attn 2 during decode steps)
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, "Qwen2_5_VLCausalLMOutputWithPast"]:
    """
    DIMV-style latent reasoning forward pass.

    Sequence layout (per sample):
        [x_txt | V | Z | Y]
         text    img  slots  answer

    Bottleneck: a custom 4D attention mask blocks Y from attending to V.
    All visual information that helps predict Y must pass through Z.
    Loss: NTP cross-entropy on Y tokens only (labels mask all other positions).

    Z is computed in parallel by LatentReasoningModule:
        X_o = concat(V, x_txt)   (all non-slot embeddings)
        Z   = LatentReasoningModule(X_o)   [B, T_v, d]
    """
    output_attentions = output_attentions if output_attentions is not None \
        else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None \
        else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # Reset per-step DIMV state
    self._last_slot_attn_weights = None

    # ── Dummy visual pass for DeepSpeed ZeRO-3 when no image in batch ────
    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device,
                                  dtype=self.model.visual.dtype)
        dummy_grid  = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + dummy_embeds.mean() * 0

    # ── Inject image patch embeddings into inputs_embeds ─────────────────
    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)

        n_img_tokens   = (input_ids == self.config.image_token_id).sum().item()
        n_img_features = image_embeds.shape[0]
        if n_img_tokens != n_img_features:
            raise ValueError(
                f"DIMV: image token count mismatch: "
                f"tokens={n_img_tokens}, features={n_img_features}"
            )

        img_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(img_mask.to(inputs_embeds.device), image_embeds)

    # ── Compute Z via LatentReasoningModule and inject at slot positions ──
    if (self.latent_reasoning is not None
            and self.slot_token_ids is not None
            and pixel_values is not None):

        # X_o = all embeddings except slot placeholders [B, N_obs, d]
        X_o, key_pad_mask = _extract_observed_context(
            inputs_embeds, input_ids, self.slot_token_ids
        )

        # Optional warm-start: last observed token hidden state
        last_hidden = (inputs_embeds[:, -1, :]
                       if self.latent_reasoning.slot_init == "last_hidden"
                       else None)

        # Parallel imputation: all T_v slots from X_o simultaneously
        Z_final, attn_weights = self.latent_reasoning(
            X_o=X_o,
            last_hidden=last_hidden,
            key_padding_mask=key_pad_mask,
        )
        # Z_final: [B, T_v, d];  attn_weights: [B, T_v, N_obs]

        self._last_slot_attn_weights = attn_weights.detach()

        # Replace [SLOT_k] placeholder embeddings with Z^(final)_k
        inputs_embeds = _inject_z_into_embeds(
            inputs_embeds, input_ids, Z_final, self.slot_token_ids
        )

        # ── DIMV-ROI: extract V_ROI* and store for L_IMP ─────────────────
        if (getattr(self, 'roi_pooler', None) is not None
                and lvr_tokens is not None
                and len(lvr_tokens) > 0):
            B = input_ids.shape[0]
            # Number of visual tokens per sample: T*(H//2)*(W//2)
            tokens_per_sample = [
                int(thw[0].item()) * (int(thw[1].item()) // 2) * (int(thw[2].item()) // 2)
                for thw in image_grid_thw
            ]
            image_embeds_split = list(torch.split(image_embeds, tokens_per_sample))

            roi_embed_list = []
            for b in range(B):
                emb_b = image_embeds_split[b] if b < len(image_embeds_split) else None
                tok = lvr_tokens[b] if b < len(lvr_tokens) else None
                v_roi_b = None
                if emb_b is not None and tok is not None:
                    if isinstance(tok, torch.Tensor) and tok.numel() > 0:
                        idx = tok.to(dtype=torch.long, device=emb_b.device)
                        v_roi_b = emb_b[idx]
                    elif hasattr(tok, '__len__') and len(tok) > 0:
                        idx = torch.tensor(list(tok), dtype=torch.long, device=emb_b.device)
                        v_roi_b = emb_b[idx]
                if v_roi_b is None or v_roi_b.shape[0] == 0:
                    v_roi_b = torch.zeros(0, image_embeds.shape[-1], device=image_embeds.device,
                                          dtype=image_embeds.dtype)
                roi_embed_list.append(v_roi_b)

            # Pad variable-K to [B, K_max, d]
            d_model = image_embeds.shape[-1]
            K_max = max(max(v.shape[0] for v in roi_embed_list), 1)
            V_roi_batch = torch.zeros(B, K_max, d_model,
                                      device=image_embeds.device, dtype=image_embeds.dtype)
            roi_pad_mask = torch.ones(B, K_max, dtype=torch.bool, device=image_embeds.device)
            for b, v in enumerate(roi_embed_list):
                if v.shape[0] > 0:
                    V_roi_batch[b, :v.shape[0]] = v
                    roi_pad_mask[b, :v.shape[0]] = False

            Z_roi_target = self.roi_pooler(V_roi=V_roi_batch, roi_padding_mask=roi_pad_mask)
            self._last_Z_final = Z_final
            self._last_Z_roi_target = Z_roi_target
        # ── END DIMV-ROI ──────────────────────────────────────────────────

    # ── Build bottleneck attention mask ───────────────────────────────────
    # Y cannot attend to V. This forces all visual information through Z.
    dimv_mask = None
    if (self.slot_token_ids is not None
            and pixel_values is not None
            and input_ids is not None):
        dimv_mask = _build_dimv_4d_mask(
            input_ids=input_ids,
            image_token_id=self.config.image_token_id,
            slot_token_ids=self.slot_token_ids,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    else:
        # No image or slots present — fall through to standard causal mask
        if attention_mask is not None:
            dimv_mask = attention_mask.to(inputs_embeds.device)

    # ── RoPE position IDs ─────────────────────────────────────────────────
    if position_ids is None:
        prefill_compiled = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if prefill_compiled or prefill_noncompiled or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            B_inner, seq_len, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_len, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, B_inner, -1)
            delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device) \
                if cache_position is not None \
                else torch.zeros((B_inner, seq_len), device=inputs_embeds.device)
            delta = delta.repeat_interleave(B_inner // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    # ── LLM forward ───────────────────────────────────────────────────────
    # Pass the bottleneck mask as attention_mask so transformers uses it
    # directly (4-D tensors are returned as-is by _preprocess_mask_arguments)
    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=dimv_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    # ── NTP loss on answer tokens only ────────────────────────────────────
    # labels: IGNORE_INDEX for x_txt + V + Z positions, real IDs for Y.
    # Slot tokens must also be masked in labels (they are by the dataset).
    #
    # Chunked cross-entropy: compute lm_head + CE in slices of the sequence
    # so peak memory is (chunk_size × vocab) instead of (full_seq × vocab).
    # At seq=16384 and vocab=152k, the full logits tensor is ~4.7 GB bf16 —
    # chunking keeps it to ~150 MB per chunk (chunk_size=128).
    loss_ntp = None
    logits = None  # not returned to caller; saves ~4.7 GB allocation

    if labels is not None:
        shift_hidden = hidden_states[..., :-1, :].contiguous()   # [B, S-1, d]
        shift_labels = labels[..., 1:].contiguous()               # [B, S-1]

        # Extra guard: mask out any slot token IDs that leaked into labels
        if self.slot_token_ids is not None:
            for sid in self.slot_token_ids:
                shift_labels = shift_labels.masked_fill(shift_labels == sid, IGNORE_INDEX)

        B, S, d = shift_hidden.shape
        flat_hidden = shift_hidden.view(B * S, d)
        flat_labels = shift_labels.view(B * S).to(flat_hidden.device)

        # Only compute CE over positions that have a real label (not IGNORE_INDEX).
        # This avoids running lm_head on the ~90% of tokens that are masked.
        active = flat_labels != IGNORE_INDEX
        if active.any():
            active_hidden = flat_hidden[active]   # [N_active, d]
            active_labels = flat_labels[active]   # [N_active]

            CHUNK = 512
            loss_fct = CrossEntropyLoss()
            total_loss = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
            n_chunks = 0
            for start in range(0, active_hidden.shape[0], CHUNK):
                chunk_h = active_hidden[start:start + CHUNK]          # [C, d]
                chunk_l = active_labels[start:start + CHUNK]          # [C]
                chunk_logits = self.lm_head(chunk_h)                  # [C, vocab]
                total_loss = total_loss + loss_fct(chunk_logits, chunk_l)
                n_chunks += 1
                del chunk_logits
            loss_ntp = total_loss / n_chunks
        else:
            # Batch has no active labels (all masked) — zero loss, still touch lm_head
            # so DeepSpeed ZeRO-3 doesn't complain about unused parameters.
            dummy = self.lm_head(flat_hidden[:1])
            loss_ntp = dummy.sum() * 0.0
    else:
        # No labels — materialise logits for generation/eval
        logits = self.lm_head(hidden_states)

    if not return_dict:
        out = (logits,) + outputs[1:]
        return (loss_ntp,) + out if loss_ntp is not None else out

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ntp,
        loss_lvr=None,
        loss_mode_switch=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=outputs.last_hidden_state[:, -1, :],
    )


def replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=False,
                                                    coconut=True,
                                                    lvr_head=True,
                                                    mode_switch_loss=False,
                                                    latent_end_token=False,
                                                    rl=False,
                                                    prototype_mode=False,
                                                    dimv_mode=False):
    
    print("#"*42)
    if dimv_mode:
        print("Activated DIMV latent reasoning mode (NTP-only loss, bottleneck mask)!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_dimv
    elif prototype_mode:
        print("Activated prototype-based parallel LVR mode!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_prototype
    elif inference_mode:
        if lvr_head:
            print("Inference mode with Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_inference
        else:
            print("Inference mode without Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_inference
    elif rl:
        print("Activated stage 2 training!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_grpo
    else:
        if latent_end_token and lvr_head:
            print("Activated latent end token mode with LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken
        elif latent_end_token and not lvr_head:
            print("Activated latent end token mode without LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_latentEndToken
        elif mode_switch_loss:
            print("Activated BCE mode swtich loss!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss
        elif lvr_head:
            print("Activated naive LVR with head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head
        else:
            print("Activated naive LVR without head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr
    
    print("#"*42)


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mode_switch: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # next_pos_lvr:Optional[bool] = False


def  set_lvr_loss_fct(loss_lvr_fct: str):
    """
        Set the loss function for LVR.
        Args:
            loss_lvr_fct (str): The type of loss function to use for LVR.
        Returns:
            A loss function object.
    """
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")

'''
    Coconut mode
    No LVR Head
'''
def qwen2_5_mixed_modality_forward_lvr(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if lvr_mode_switch:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []

                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = _get_rope_index_compat(
                    self.model, input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask, self.config,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    No LVR Head
'''
def qwen2_5_mixed_modality_forward_lvr_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if last_position_hidden_state is not None:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    # if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
    #     # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
    #     dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
    #     dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
    #     dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
    #     image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
    #     # Operates as maksed_scatter for the image tokens
    #     # However the values are all zeros so it dosen't affect the embeddings.
    #     # This could avoid deepspeed error when some batch only has texts.
    #     inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []

                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = _get_rope_index_compat(
                    self.model, input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask, self.config,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )



'''
    Coconut mode;
    LVR head;
    Note that this forward function is used for inferencing all the LVR models with a LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )





def qwen2_5_mixed_modality_forward_lvr_with_head_inference(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            # selected_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start]
            # lvr_head_output = self.lvr_head(selected_hidden_states)
            # outputs.last_hidden_state[batch_indices, seq_positions_start] = lvr_head_output

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

        # mode switch loss

        lvr_or_lvrstart_mask = (input_ids == self.config.lvr_start_id) | (input_ids == self.config.lvr_id)

        # Find the next tokens of each position
        shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=1)
        # the lvr token that is right before lvr_end token
        is_last_lvr = lvr_or_lvrstart_mask & (shifted_input_ids == self.config.lvr_end_id)
        # 1 if it's the last <lvr> before <lvr_end>, else 0
        targets = is_last_lvr.float()  # [batch_size, seq_len]

        lvr_end_logits = logits[..., self.config.lvr_end_id]  # [batch_size, seq_len]

        # Apply mask to focus only on <lvr_start>,<lvr> token positions
        masked_logits = lvr_end_logits[lvr_or_lvrstart_mask]  # [num_lvr_tokens]
        masked_targets = targets[lvr_or_lvrstart_mask]        # [num_lvr_tokens]

        loss_mode_switch = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            '''In this mode, <|lvr_latent_end|> is also a latent token'''
            seq_positions_start_latentend = seq_positions_latentend - 1
            outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend] = self.lvr_head(outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend])


    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))
    mode_switch_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_mode_switch_fct', 'bce'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )



'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_latentEndToken(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))
    mode_switch_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_mode_switch_fct', 'bce'))

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


"""
    Forward function for stage 2 RL
    Kinda messy since in this stage, the transofmers will be 4.51.3 < 4.54 in stage I
    Will fix this inconsistency in final release
"""
def qwen2_5_mixed_modality_forward_lvr_rl(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''In this mode, no lvr_tokens'''
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch].to(inputs_embeds.device)
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(_extract_image_embeds(image_embeds), dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = _get_rope_index_compat(
                self.model, input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask, self.config,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # check if there is lvr_head
    if self.config.lvr_head:
        '''apply lvr_head in _inference mode'''
        if lvr_mode_switch is not None:
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_lvr = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # No lvr loss in this mode
        loss_lvr = None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )







'''Liger kernel'''
# def qwen2_5_mixed_modality_forward_lvr_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
#     lvr_tokens: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
#             '''
#                 Filling the lvr tokens with image embeddings.
#                 Applicable when each image input has multiple bboxes
#             '''
#             total_tokens = torch.sum(mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
#             batch_size = input_ids.size(0) 
#             # lvr mask for lvr token locations in the batch, [bs, seq_length]
#             # in each instance, lvr tokens are True, others are False
#             lvr_mask = input_ids == self.config.lvr_id  
#             # Total length = number of <lvr> tokens in the batch
#             # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
#             batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

#            #  GLOBAL starting index in `image_embeds` of each image in the batch
#             image_token_offsets = torch.cumsum(
#                 F.pad(total_tokens, (1, 0)), dim=0
#             )[:-1]  # shape [B], offset into image_embeds for each batch element

#             global_lvr_token_indices = []

#             for b, lvr_ids in enumerate(lvr_tokens):
#                 # Convert local to global index
#                 offset = image_token_offsets[b].item()
#                 global_lvr_token_indices.append(lvr_ids + offset)
#             global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

#             # Step 3: Gather the selected visual embeddings
#             selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

#             # Step 4: Replace in input_embeds at the right batch and position
#             # Prepare indexing
#             # replaced_embeds = inputs_embeds.clone()
#             inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds



#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
#             video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]

#     lvr_loss_fct = set_lvr_loss_fct(getattr(self.config, 'loss_lvr_fct', 'mse'))


#     loss = None
#     loss_ce = None
#     loss_lvr = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)
#         # Don't want CE loss for <lvr> token
#         shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss_ce = lce(self.lm_head.weight, shift_hidden_states, shift_labels)

        
#         # lvr loss
#         # Get last hidden states for <lvr> token positions
#         seq_positions_start = seq_positions - 1  # Now points to lvr_start
#         selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#         # Compute LVR loss between predicted and inserted lvr embeddings
#         loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
#             shift_labels = shift_labels.view(-1)
#             # Don't want CE loss for <lvr> token
#             shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss_ce = loss_fct(shift_logits, shift_labels)

#             # lvr loss
#             # Get last hidden states for <lvr> token positions
#             seq_positions_start = seq_positions - 1  # Now points to lvr_start
#             selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#             # Compute LVR loss between predicted and inserted lvr embeddings
#             loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         loss_ce=loss_ce,
#         loss_lvr=loss_lvr,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )
