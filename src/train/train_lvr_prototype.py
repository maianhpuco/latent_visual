"""
Prototype-based parallel LVR training entry point.

Replaces sequential [lvr] token generation with K independent prototype slots:
    z_k = g_φ(p_k, O)   for all k simultaneously, ∂z_k/∂z_j = 0

Based on train_lvr_lora.py but activates prototype modules instead of
sequential LVR injection.
"""
import sys
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel, TrainerCallback

from src.model.qwen_lvr_model import QwenWithLVR
from src.config.prototype_lvr_config import PrototypeLVRConfig
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_supervised_data_module_lvr, make_packed_supervised_data_module_lvr, make_packed_supervised_data_module_lvr_fixedToken
from src.params import DataArguments, ModelArguments, TrainingArguments

from train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

try:
    from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
except ImportError:
    OCIFolderCheckpointHandler = None
    create_temp_dir = None
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None


class TimeEstimationCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        self._last_printed_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        if state.max_steps > 0 and state.global_step > 0 and state.global_step != self._last_printed_step:
            elapsed = time.time() - self._start_time
            progress = state.global_step / state.max_steps
            eta_seconds = elapsed / progress - elapsed
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            gpu_info = ""
            try:
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
                    gpu_info = f" | GPU: {used:.1f}/{total:.0f}GB"
            except Exception:
                pass
            self._last_printed_step = state.global_step
            print(
                f"[Time] Step {state.global_step}/{state.max_steps} "
                f"({100*progress:.1f}%) | "
                f"Elapsed: {elapsed/3600:.2f}h | "
                f"ETA: {eta_h}h {eta_m}m"
                f"{gpu_info}",
                flush=True,
            )


class PrototypeWarmupCallback(TrainerCallback):
    """
    During steps 0 → warmup_steps_prototype_only, only prototype modules train:
      - LoRA mode:   freeze LoRA adapter weights
      - Full FT mode: freeze LLM (lm_head + model) weights
    After warmup, the frozen weights are unfrozen.
    """

    def __init__(self, warmup_steps: int, lora_enable: bool = False):
        self.warmup_steps = warmup_steps
        self.lora_enable = lora_enable
        self._frozen = False

    def _prototype_only(self, name: str) -> bool:
        return "prototype_bank" in name or "prototype_cross_attn" in name

    def _set_non_proto_requires_grad(self, model, requires_grad: bool):
        if self.lora_enable:
            # LoRA mode: only touch LoRA adapter weights
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = requires_grad
        else:
            # Full FT mode: freeze/unfreeze the LLM (everything except prototype modules
            # and the frozen vision tower / merger which are already requires_grad=False)
            for name, param in model.named_parameters():
                if not self._prototype_only(name):
                    param.requires_grad = requires_grad

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps > 0:
            self._set_non_proto_requires_grad(model, False)
            self._frozen = True
            mode = "LoRA adapters" if self.lora_enable else "LLM weights"
            print(f"[PrototypeWarmup] Froze {mode} for first {self.warmup_steps} steps. "
                  "Training prototype_bank and prototype_cross_attn only.")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if self._frozen and state.global_step >= self.warmup_steps:
            self._set_non_proto_requires_grad(model, True)
            self._frozen = False
            mode = "LoRA adapters" if self.lora_enable else "LLM weights"
            print(f"[PrototypeWarmup] Step {state.global_step}: Unfroze {mode}.")


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def find_target_linear_names(model, lora_namespan_exclude=None, num_lora_modules=-1):
    """Find all linear layers in the LLM backbone (exclude vision tower)."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = ["visual", "lm_head"]
    linear_cls = torch.nn.Linear
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    rank0_print(f"[LoRA] Targeting {len(lora_module_names)} modules")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    # For PEFT-wrapped models unwrap to the underlying HF model first.
    # Do NOT use hasattr(model, 'base_model'): HuggingFace PreTrainedModel defines
    # base_model as a property that returns the inner backbone (e.g. Qwen2_5_VLModel),
    # not self — so the old check fired even in full-FT mode and broke the lookup.
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except ImportError:
        is_peft = False

    if is_peft:
        base = model.base_model.model
    else:
        base = model

    if hasattr(base, 'visual'):
        vision_tower = base.visual
    elif hasattr(base, 'model') and hasattr(base.model, 'visual'):
        vision_tower = base.model.visual
    else:
        raise AttributeError(f"Cannot find vision tower in {type(base)}")

    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(vision_tower.merger.parameters(), not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model_pth = training_args.checkpoint_name if training_args.checkpoint_name else model_args.model_id

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    config.latent_end_token = False
    config.lvr_head = False
    config.lvr_head_type = model_args.lvr_head_type

    # Build PrototypeLVRConfig and attach to model config
    proto_cfg = PrototypeLVRConfig(
        num_prototypes=model_args.num_prototypes,
        prototype_num_heads=model_args.prototype_num_heads,
        prototype_dropout=model_args.prototype_dropout,
        loss_diversity_lambda=training_args.loss_diversity_lambda,
        loss_focus_lambda=training_args.loss_focus_lambda,
        warmup_steps_prototype_only=training_args.warmup_steps_prototype_only,
    )
    config.prototype_config = proto_cfg

    if "Qwen2.5" in model_args.model_id:
        # Activate prototype-mode forward
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            prototype_mode=True,
        )
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
        )
        replace_qwen_2_5_vl_patch_emb()
    else:
        raise ValueError("Only Qwen2.5 supported.")

    model.config.use_cache = False

    if training_args.lora_enable:
        # ── LoRA fine-tuning ───────────────────────────────────────────────────
        rank0_print("Applying LoRA...")
        # Exclude vision tower, lm_head, AND prototype modules from LoRA targeting
        lora_namespan_exclude = ["visual", "lm_head", "prototype_bank", "prototype_cross_attn"]
        target_modules = find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude)

        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Freeze vision tower (peft unfreezes everything by default)
        configure_vision_tower(model, training_args, compute_dtype, training_args.device)

        # Always keep prototype modules trainable (peft may have frozen them)
        for name, param in model.named_parameters():
            if "prototype_bank" in name or "prototype_cross_attn" in name:
                param.requires_grad = True
    else:
        # ── Full fine-tuning ───────────────────────────────────────────────────
        rank0_print("Full fine-tuning mode (no LoRA)...")
        configure_vision_tower(model, training_args, compute_dtype, training_args.device)
        configure_llm(model, training_args)

        # Prototype modules are always trainable
        for name, param in model.named_parameters():
            if "prototype_bank" in name or "prototype_cross_attn" in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        # Under DeepSpeed ZeRO-3 sharding, non-rank-0 processes see numel()==0
        pct = (100 * trainable / total) if total > 0 else float("nan")
        rank0_print(f"[Full FT] Trainable: {trainable:,} / {total:,} ({pct:.1f}%)")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    # Register original LVR tokens (needed for dataset pipeline compat)
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

    lvr_id            = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    lvr_start_id      = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    lvr_end_id        = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")

    model.config.lvr_id            = lvr_id
    model.config.lvr_latent_end_id = lvr_latent_end_id
    model.config.lvr_start_id      = lvr_start_id
    model.config.lvr_end_id        = lvr_end_id

    # Register prototype tokens [proto_0]...[proto_{K-1}]
    # For PEFT-wrapped models unwrap to the underlying HF model; for full FT model IS the base model.
    # Do NOT use hasattr(model, 'base_model'): HF PreTrainedModel.base_model returns the inner
    # backbone (Qwen2_5_VLModel), not self, so the check fires in full-FT mode and breaks lookup.
    try:
        from peft import PeftModel as _PeftModel
        _is_peft = isinstance(model, _PeftModel)
    except ImportError:
        _is_peft = False
    base_model = model.base_model.model if _is_peft else model
    base_model.setup_proto_tokens(processor.tokenizer)
    # Sync proto_token_ids onto wrapper (no-op for full FT since model == base_model)
    model.proto_token_ids = base_model.proto_token_ids

    # Sync vocab size
    _vocab_size = getattr(model.config, 'vocab_size', None)
    if _vocab_size is None:
        _text_cfg = getattr(model.config, 'text_config', None)
        _vocab_size = getattr(_text_cfg, 'vocab_size', 0) if _text_cfg else 0
    if _vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    model.config.loss_lvr_fct = training_args.loss_lvr_fct

    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        if model_args.max_lvr_tokens is not None:
            data_module, total_data_len = make_packed_supervised_data_module_lvr_fixedToken(
                model_id=model_args.model_id, processor=processor,
                max_lvr_tokens=model_args.max_lvr_tokens, data_args=data_args,
                training_args=training_args, latent_end_token=False)
        else:
            data_module, total_data_len = make_packed_supervised_data_module_lvr(
                model_id=model_args.model_id, processor=processor,
                data_args=data_args, training_args=training_args,
                latent_end_token=False)
        if not training_args.max_steps:
            training_args.max_steps = total_data_len // (
                training_args.gradient_accumulation_steps
                * training_args.world_size
                * training_args.per_device_train_batch_size)
        replace_train_dataloader()
    else:
        data_module = make_supervised_data_module_lvr(
            model_id=model_args.model_id, processor=processor,
            data_args=data_args, latent_end_token=False)

    callbacks = [TimeEstimationCallback()]
    if training_args.warmup_steps_prototype_only > 0:
        callbacks.append(PrototypeWarmupCallback(
            warmup_steps=training_args.warmup_steps_prototype_only,
            lora_enable=training_args.lora_enable,
        ))

    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=None,
        oci_handler=None,
        callbacks=callbacks,
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
