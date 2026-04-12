# from .dpo_trainer import QwenDPOTrainer
from .sft_trainer import QwenSFTTrainer
from .lvr_trainer import QwenLVRSFTTrainer

try:
    from .grpo_trainer import QwenGRPOTrainer
except ImportError:
    QwenGRPOTrainer = None  # trl not installed; only needed for stage-2

__all__ = ["QwenSFTTrainer", "QwenLVRSFTTrainer", "QwenGRPOTrainer"]