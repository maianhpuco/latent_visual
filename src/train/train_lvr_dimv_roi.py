"""
DIMV-ROI training entry point.

Identical to train_lvr.py DIMV path but:
- Forces dimv_mode=True and use_roi_supervision=True.
- Uses make_packed_supervised_data_module_dimv_roi (reads bboxes → lvr_tokens).
- Passes ROI imputation fields to LatentReasoningConfig.
"""

import dataclasses
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, TrainerCallback

from src.config.latent_reasoning_config import LatentReasoningConfig
from src.dataset import make_packed_supervised_data_module_dimv_roi
from src.eval.vstar_validator import VStarValidator
from src.model.qwen_lvr_model import QwenWithLVR
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.trainer import QwenLVRSFTTrainer
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

from train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

try:
    from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
except ImportError:
    OCIFolderCheckpointHandler = None
    create_temp_dir = None

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
                f"({100*progress:.1f}%) | Elapsed: {elapsed/3600:.2f}h | "
                f"ETA: {eta_h}h {eta_m}m{gpu_info}",
                flush=True,
            )


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


def parse_int_list(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return []
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    if isinstance(raw_value, (list, tuple)):
        return [int(x) for x in raw_value]
    return []


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    # Force DIMV-ROI mode regardless of CLI flags
    model_args.dimv_mode = True
    training_args.use_roi_supervision = True

    if not training_args.checkpoint_dir_roi:
        training_args.checkpoint_dir_roi = training_args.output_dir

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # OCI checkpoint handler
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint and OCIFolderCheckpointHandler is not None:
        oci_handler = OCIFolderCheckpointHandler(training_args.checkpoint_name)
        temp_folder = create_temp_dir(
            prefix=os.path.join(training_args.output_dir, "tmp_ckpt_")
        )

    # Resume from checkpoint
    model_pth = model_args.model_id
    if (
        training_args.resume_from_checkpoint
        and isinstance(training_args.resume_from_checkpoint, str)
        and os.path.isdir(training_args.resume_from_checkpoint)
    ):
        model_pth = training_args.resume_from_checkpoint

    # Build LatentReasoningConfig with ROI fields
    early_checkpoint_steps = parse_int_list(training_args.early_checkpoint_steps)
    lr_cfg = LatentReasoningConfig(
        num_reasoning_slots=model_args.num_reasoning_slots,
        slot_init=model_args.slot_init,
        num_refinement_steps=model_args.num_refinement_steps,
        num_attn_heads=model_args.dimv_num_attn_heads,
        use_roi_supervision=True,
        roi_pool_method=training_args.roi_pool_method,
        imputation_loss_type=training_args.imputation_loss_type,
        imputation_loss_lambda=training_args.imputation_loss_lambda,
        nce_temperature=training_args.nce_temperature,
        checkpoint_dir=training_args.checkpoint_dir_roi,
        vstar_val_fraction=training_args.vstar_val_fraction,
        validate_every_n_steps=training_args.validate_every_n_steps,
        early_checkpoint_steps=early_checkpoint_steps,
        vstar_val_seed=training_args.vstar_val_seed,
    )

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    config.latent_reasoning_config = dataclasses.asdict(lr_cfg)
    config.use_cache = False

    replace_qwen2_5_with_mixed_modality_forward_lvr(
        coconut=model_args.coconut,
        lvr_head=model_args.lvr_head,
        mode_switch_loss=training_args.mode_switch_loss,
        latent_end_token=model_args.latent_end_token,
        dimv_mode=True,
    )

    model = QwenWithLVR.from_pretrained(
        model_pth,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
        ),
    )

    replace_qwen_2_5_vl_patch_emb()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )

    for special_tok in ("<|lvr_start|>", "<|lvr|>", "<|lvr_latent_end|>", "<|lvr_end|>"):
        processor.tokenizer.add_tokens(special_tok, special_tokens=True)

    model.config.lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    model.config.lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")

    model.setup_slot_tokens(processor.tokenizer)
    rank0_print(f"[DIMV-ROI] Slot tokens registered: {model.slot_token_ids[:4]}…")

    _vocab_size = getattr(model.config, "vocab_size", None)
    if _vocab_size is None:
        _text_cfg = getattr(model.config, "text_config", None)
        _vocab_size = getattr(_text_cfg, "vocab_size", 0) if _text_cfg else 0
    if _vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    model.config.loss_lvr_fct = training_args.loss_lvr_fct

    # Data module
    assert training_args.enable_data_packing, (
        "DIMV-ROI training requires --enable_data_packing True"
    )
    training_args.per_device_train_batch_size = 1

    slot_token_ids = list(model.slot_token_ids)
    data_module, total_data_len = make_packed_supervised_data_module_dimv_roi(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        training_args=training_args,
        slot_token_ids=slot_token_ids,
    )

    if not training_args.max_steps:
        training_args.max_steps = total_data_len // (
            training_args.gradient_accumulation_steps
            * training_args.world_size
            * training_args.per_device_train_batch_size
        )

    replace_train_dataloader()

    # Create fixed V* validation split (rank 0 only)
    if training_args.validate_every_n_steps > 0 and local_rank in (None, -1, 0, "0"):
        val_indices = VStarValidator.create_fixed_val_set(
            val_fraction=training_args.vstar_val_fraction,
            seed=training_args.vstar_val_seed,
            save_path=os.path.join(
                training_args.checkpoint_dir_roi,
                "vstar_val_indices.json",
            ),
            configs_dir=getattr(training_args, "vstar_configs_dir", None),
        )
        rank0_print(
            f"[DIMV-ROI] Fixed V* validation split ready: {len(val_indices)} samples."
        )

    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        callbacks=[TimeEstimationCallback()],
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
