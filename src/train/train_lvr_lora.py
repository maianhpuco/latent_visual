import sys
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel, TrainerCallback

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_supervised_data_module_lvr, make_packed_supervised_data_module_lvr, make_packed_supervised_data_module_lvr_fixedToken, make_packed_supervised_data_module_dimv
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
from src.config.latent_reasoning_config import LatentReasoningConfig

local_rank = None


class TimeEstimationCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        self._last_printed_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only print from rank 0, and only once per step (on_log fires per log-dict)
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
                import torch
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
    # Unwrap PEFT wrapper if present (PeftModel wraps the base model)
    base = model
    if hasattr(model, 'base_model'):
        base = model.base_model.model  # PeftModel -> LoraModel -> original model

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
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model_pth = training_args.checkpoint_name if training_args.checkpoint_name else model_args.model_id

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head
    config.lvr_head_type = model_args.lvr_head_type

    # DIMV: store as plain dict so transformers can JSON-serialize it in config.json.
    # _init_latent_reasoning accepts both dict and LatentReasoningConfig.
    if model_args.dimv_mode:
        import dataclasses
        config.latent_reasoning_config = dataclasses.asdict(LatentReasoningConfig(
            num_reasoning_slots=model_args.num_reasoning_slots,
            slot_init=model_args.slot_init,
            num_refinement_steps=model_args.num_refinement_steps,
            num_attn_heads=model_args.dimv_num_attn_heads,
        ))

    if "Qwen2.5" in model_args.model_id:
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            coconut=model_args.coconut,
            lvr_head=model_args.lvr_head,
            mode_switch_loss=training_args.mode_switch_loss,
            latent_end_token=model_args.latent_end_token,
            dimv_mode=model_args.dimv_mode,
        )
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
        )
        if model_args.lvr_head:
            model._init_lvr_head(lvr_head_type=model_args.lvr_head_type)
        if model_args.latent_end_token:
            model._init_lvr_latent_end_emb()
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        replace_qwen_2_5_vl_patch_emb()
    else:
        raise ValueError("Only Qwen2.5 supported.")

    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    rank0_print("Applying LoRA...")
    # In DIMV mode, exclude latent_reasoning from LoRA — it is trained fully.
    lora_namespan_exclude = ["visual", "lm_head"]
    if model_args.dimv_mode:
        lora_namespan_exclude.append("latent_reasoning")
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

    # In DIMV mode, latent_reasoning params were frozen by PEFT — unfreeze them.
    if model_args.dimv_mode:
        for name, param in model.named_parameters():
            if "latent_reasoning" in name:
                param.requires_grad = True
        rank0_print(
            f"[DIMV] latent_reasoning params unfrozen: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'latent_reasoning' in n):,}"
        )

    # Freeze vision tower (peft unfreezes everything by default)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    # ──────────────────────────────────────────────────────────────────────────

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    processor.tokenizer.add_tokens("<|lvr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|lvr_end|>", special_tokens=True)

    lvr_id           = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    lvr_start_id     = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    lvr_end_id       = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")

    model.config.lvr_id           = lvr_id
    model.config.lvr_latent_end_id = lvr_latent_end_id
    model.config.lvr_start_id     = lvr_start_id
    model.config.lvr_end_id       = lvr_end_id

    # DIMV: add [SLOT_0]…[SLOT_{T_v-1}] tokens and resize embeddings.
    # model.setup_slot_tokens is resolved via PEFT's __getattr__ chain to QwenWithLVR.
    if model_args.dimv_mode:
        model.setup_slot_tokens(processor.tokenizer)
        rank0_print(f"[DIMV] Slot tokens registered: {model.slot_token_ids[:4]}…")

    _vocab_size = getattr(model.config, 'vocab_size', None)
    if _vocab_size is None:
        _text_cfg = getattr(model.config, 'text_config', None)
        _vocab_size = getattr(_text_cfg, 'vocab_size', 0) if _text_cfg else 0
    if _vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    model.config.loss_lvr_fct = training_args.loss_lvr_fct

    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1

        if model_args.dimv_mode:
            # DIMV always uses the DIMV dataset — slot_token_ids come from the model
            slot_token_ids = list(model.slot_token_ids)
            data_module, total_data_len = make_packed_supervised_data_module_dimv(
                model_id=model_args.model_id,
                processor=processor,
                data_args=data_args,
                training_args=training_args,
                slot_token_ids=slot_token_ids,
            )
        elif model_args.max_lvr_tokens is not None:
            data_module, total_data_len = make_packed_supervised_data_module_lvr_fixedToken(
                model_id=model_args.model_id, processor=processor,
                max_lvr_tokens=model_args.max_lvr_tokens, data_args=data_args,
                training_args=training_args, latent_end_token=model_args.latent_end_token)
        else:
            data_module, total_data_len = make_packed_supervised_data_module_lvr(
                model_id=model_args.model_id, processor=processor,
                data_args=data_args, training_args=training_args,
                latent_end_token=model_args.latent_end_token)
        if not training_args.max_steps:
            training_args.max_steps = total_data_len // (
                training_args.gradient_accumulation_steps
                * training_args.world_size
                * training_args.per_device_train_batch_size)
        replace_train_dataloader()
    else:
        data_module = make_supervised_data_module_lvr(
            model_id=model_args.model_id, processor=processor,
            data_args=data_args, latent_end_token=model_args.latent_end_token)

    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=None,
        oci_handler=None,
        callbacks=[TimeEstimationCallback()],
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
