import os
import json
import torch
import torch.nn as nn
import wandb
from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

class DIMVCheckpointCallback(TrainerCallback):
    """
    Saves scheduled DIMV checkpoints and runs V* validation on rank 0.
    """
    def __init__(self, trainer_ref):
        self.trainer_ref = trainer_ref

    def on_step_end(self, args, state, control, **kwargs):
        if getattr(self.trainer_ref.model, 'latent_reasoning', None) is None:
            return
        if getattr(args, 'validate_every_n_steps', 0) <= 0:
            return
        if not getattr(args, 'should_save', False):
            return
        step = state.global_step
        if self.trainer_ref._should_checkpoint(step):
            ckpt_path = self.trainer_ref._save_dimv_checkpoint(step)
            self.trainer_ref._validate_on_vstar(step, ckpt_path)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenLVRSFTTrainer(Trainer):

    def __init__(self, *args, temp_folder=None, oci_handler=None, **kwargs):
        super(QwenLVRSFTTrainer, self).__init__(*args, **kwargs)
        # This trainer implements its own compute_loss and does not use
        # Hugging Face's num_items_in_batch token-count scaling. Mark that
        # explicitly so Trainer.get_batch_samples does not try to gather a CPU
        # scalar before packed batches are moved onto CUDA.
        self.model_accepts_loss_kwargs = False
        if getattr(self.args, "average_tokens_across_devices", False):
            logger.info(
                "[LVRTrainer] Disabling average_tokens_across_devices because "
                "custom compute_loss ignores num_items_in_batch."
            )
            self.args.average_tokens_across_devices = False
        # if online checkpointing
        if oci_handler:
            self.oci_handler = oci_handler
            self.temp_folder = temp_folder     # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
        # DIMV-ROI: register checkpoint/validation callback
        self._last_logged_loss = 0.0
        self._last_logged_loss_ntp = 0.0
        self._last_logged_loss_imp = 0.0
        if (
            getattr(self.model, 'latent_reasoning', None) is not None
            and getattr(self.args, 'validate_every_n_steps', 0) > 0
        ):
            self.add_callback(DIMVCheckpointCallback(self))

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            lvr_head_parameters =[]

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.lvr_head_lr is not None:
                lr_mapper["lvr_head"] = self.args.lvr_head_lr
                lvr_head_parameters = [name for name, _ in opt_model.named_parameters() if "lvr_head" in name]

            prototype_parameters = []
            if getattr(self.args, 'prototype_lr', None) is not None:
                lr_mapper["prototype"] = self.args.prototype_lr
                prototype_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "prototype_bank" in name or "prototype_cross_attn" in name
                ]

            latent_reasoning_parameters = []
            if getattr(self.args, 'latent_reasoning_lr', None) is not None:
                lr_mapper["latent_reasoning"] = self.args.latent_reasoning_lr
                latent_reasoning_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "latent_reasoning" in name or "roi_pooler" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = (
                    merger_parameters + visual_parameters + lvr_head_parameters
                    + prototype_parameters + latent_reasoning_parameters
                )
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
                if lvr_head_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.lvr_head_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.lvr_head_lr,
                            },
                        ]
                    )

                if prototype_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in prototype_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.prototype_lr,
                                "name": "prototype",
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in prototype_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.prototype_lr,
                                "name": "prototype_no_decay",
                            },
                        ]
                    )

                if latent_reasoning_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in latent_reasoning_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.latent_reasoning_lr,
                                "name": "latent_reasoning",
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in latent_reasoning_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.latent_reasoning_lr,
                                "name": "latent_reasoning_no_decay",
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # output_dir is local; now we save to cloud if needed
        if getattr(self, 'temp_folder', None):
            remote_chkpt_folder = os.path.join(self.args.remote_output_dir,checkpoint_folder)
            if remote_chkpt_folder[0] == '/':
                remote_chkpt_folder = remote_chkpt_folder[1:]       #remote pathing rules will take bucket//checkpoints, need to remove the dup
            self.oci_handler.save_checkpoint(output_dir,remote_chkpt_folder)    #save local chkpt to remote folder
            # remove the local 
            self.temp_folder.cleanup(checkpoint_name=checkpoint_folder)


        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(self, model, inputs,num_items_in_batch=None, return_outputs=False):

        if self.args.enable_data_packing:
            batch_size = inputs['input_ids'].size(0)
            total_tokens = inputs['input_ids'].size(0) * inputs['input_ids'].size(1)
            self.log({
            "batch_size": batch_size,
            "tokens_per_device": total_tokens,})

        outputs = model(**inputs)
        loss_ce = outputs.loss_ce
        loss_lvr = outputs.loss_lvr
        loss_mode_switch = outputs.loss_mode_switch

        if self.args.mode_switch_loss:
            loss = loss_ce + self.args.loss_lvr_lambda * loss_lvr + self.args.loss_mode_switch_lambda * loss_mode_switch
        else:
            # Guard: loss_lvr is None when prototype_mode forward finds no proto tokens in batch
            if self.args.loss_lvr_lambda > 0 and loss_lvr is not None:
                loss = loss_ce + self.args.loss_lvr_lambda * loss_lvr
            else:
                loss = loss_ce

        # ── DIMV-ROI: add imputation loss ─────────────────────────────────
        loss_imp = torch.tensor(0.0, device=loss_ce.device)
        if (getattr(self.args, 'use_roi_supervision', False)
                and getattr(model, 'roi_pooler', None) is not None
                and getattr(model, '_last_Z_final', None) is not None
                and getattr(model, '_last_Z_roi_target', None) is not None):
            from src.model.latent_reasoning_module import compute_imputation_loss
            loss_imp = compute_imputation_loss(
                Z_final=model._last_Z_final,
                Z_roi_target=model._last_Z_roi_target,
                loss_type=getattr(self.args, 'imputation_loss_type', 'cosine'),
                temperature=getattr(self.args, 'nce_temperature', 0.07),
            )
            lam = getattr(self.args, 'imputation_loss_lambda', 0.1)
            loss = loss + lam * loss_imp
        # ── END DIMV-ROI ──────────────────────────────────────────────────

        # Log each component
        log_dict = {
            "loss_total": loss.detach().item(),
            "loss_ce": loss_ce.detach().item(),
            "loss_lvr": loss_lvr.detach().item() if loss_lvr is not None else 0.0,
            "loss_mode_switch": loss_mode_switch.detach().item() if loss_mode_switch is not None else 0.0,
            "loss_imp": loss_imp.detach().item(),
        }
        self._last_logged_loss = loss.detach().item()
        self._last_logged_loss_ntp = loss_ce.detach().item()
        self._last_logged_loss_imp = loss_imp.detach().item()

        # Log DIMV slot attention entropy (collapse detection for latent reasoning slots)
        _slot_A = getattr(model, '_last_slot_attn_weights', None)
        if _slot_A is not None:
            import torch.nn.functional as _F
            _eps = 1e-9
            _slot_entropy = -(_slot_A.float() * (_slot_A.float() + _eps).log()).sum(dim=-1).mean().item()
            log_dict["dimv_slot_attn_entropy"] = _slot_entropy

        # Log prototype-specific diagnostics (slot collapse detection)
        _Z = getattr(model, '_last_proto_Z', None)
        _A = getattr(model, '_last_proto_attn_weights', None)
        if _Z is not None and _A is not None:
            import torch.nn.functional as _F
            _Z_norm = _F.normalize(_Z.float(), dim=-1)
            _sim = torch.bmm(_Z_norm, _Z_norm.transpose(1, 2))
            _K = _Z.shape[1]
            _off_diag = ~torch.eye(_K, dtype=torch.bool, device=_Z.device)
            _mean_cos = _sim[:, _off_diag].abs().mean().item()
            _eps = 1e-9
            _entropy = -(_A.float() * (_A.float() + _eps).log()).sum(dim=-1).mean().item()
            log_dict["mean_cosine_sim"] = _mean_cos
            log_dict["mean_proto_entropy"] = _entropy
            if _mean_cos > 0.8:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"[Prototype] Slot collapse detected! mean_cosine_sim={_mean_cos:.3f}. "
                    "Consider increasing loss_diversity_lambda or reducing prototype_lr."
                )

        self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    # ── DIMV-ROI checkpoint and validation helpers ────────────────────────

    def _should_checkpoint(self, step: int) -> bool:
        early_raw = getattr(self.args, 'early_checkpoint_steps', '10,100') or '10,100'
        if isinstance(early_raw, str):
            early_steps = [int(s) for s in early_raw.split(',') if s.strip()]
        elif isinstance(early_raw, (list, tuple)):
            early_steps = [int(s) for s in early_raw]
        else:
            early_steps = [10, 100]
        validate_every = getattr(self.args, 'validate_every_n_steps', 500)
        max_steps = getattr(self.args, 'max_steps', 2500)
        return step in early_steps or (step > 0 and step % validate_every == 0) or step == max_steps

    def _save_dimv_checkpoint(self, step: int) -> str:
        ckpt_path = os.path.join(self._get_output_dir(trial=None), f"{PREFIX_CHECKPOINT_DIR}-{step}")
        if os.path.isdir(ckpt_path):
            print(f"[DIMV] Checkpoint already exists: {ckpt_path}")
            return ckpt_path
        self._save_checkpoint(self.model, trial=None)
        print(f"[DIMV] Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def _validate_on_vstar(self, step: int, ckpt_path: str) -> dict:
        checkpoint_dir = getattr(self.args, 'checkpoint_dir_roi', None) or self.args.output_dir
        val_indices_path = os.path.join(checkpoint_dir, "vstar_val_indices.json")
        val_results_dir = os.path.join(checkpoint_dir, "val_results")
        os.makedirs(val_results_dir, exist_ok=True)

        try:
            from src.eval.vstar_validator import VStarValidator
        except ImportError:
            print("[DIMV] VStarValidator not found. Skipping validation.")
            return {}

        if os.path.exists(val_indices_path):
            with open(val_indices_path, "r") as f:
                val_indices = json.load(f)
        else:
            val_indices = VStarValidator.create_fixed_val_set(
                val_fraction=getattr(self.args, 'vstar_val_fraction', 0.30),
                seed=getattr(self.args, 'vstar_val_seed', 42),
                save_path=val_indices_path,
                configs_dir=getattr(self.args, 'vstar_configs_dir', None),
            )

        print(f"[DIMV] Running v*star validation: {len(val_indices)} samples at step {step} ...")

        device = next(self.model.parameters()).device
        validator = VStarValidator(
            model=self.model,
            processor=getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None),
            val_indices=val_indices,
            device=device,
            configs_dir=getattr(self.args, 'vstar_configs_dir', None),
            max_new_tokens=getattr(self.args, 'vstar_max_new_tokens', 32),
        )
        try:
            metrics = validator.evaluate()
        except NotImplementedError:
            print("[DIMV] VStarValidator._eval_batch() not implemented. Skipping validation.")
            metrics = {"accuracy": 0.0, "n_correct": 0, "n_samples": len(val_indices), "per_sample": []}

        result_record = {
            "step": step,
            "checkpoint_path": ckpt_path,
            "train_loss": self._last_logged_loss,
            "train_loss_ntp": self._last_logged_loss_ntp,
            "train_loss_imp": self._last_logged_loss_imp,
            "val_accuracy": metrics.get("accuracy", 0.0),
            "val_n_correct": metrics.get("n_correct", 0),
            "val_n_samples": metrics.get("n_samples", 0),
            "per_sample": metrics.get("per_sample", []),
        }
        result_path = os.path.join(val_results_dir, f"step_{step}.json")
        with open(result_path, "w") as f:
            json.dump(result_record, f, indent=2, ensure_ascii=False)

        self.log({
            "val/accuracy": metrics.get("accuracy", 0.0),
            "val/n_correct": metrics.get("n_correct", 0),
            "val/n_samples": metrics.get("n_samples", 0),
        })
        print(f"[DIMV] Step {step} | val_accuracy={metrics.get('accuracy', 0):.4f}")
        print(f"[DIMV] Results saved → {result_path}")
        return metrics
