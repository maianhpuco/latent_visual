from .lvr_sft_dataset import make_supervised_data_module_lvr
from .lvr_sft_dataset_packed import (
    make_packed_supervised_data_module_lvr,
    make_packed_supervised_data_module_dimv,
    make_packed_supervised_data_module_dimv_roi,
)

try:
    from .lvr_sft_dataset_packed_fixedToken import make_packed_supervised_data_module_lvr_fixedToken
except ImportError:
    make_packed_supervised_data_module_lvr_fixedToken = None  # file not present; only needed when max_lvr_tokens is set

try:
    from .sft_dataset import make_supervised_data_module
except ImportError:
    make_supervised_data_module = None

try:
    from .dpo_dataset import make_dpo_data_module
    from .grpo_dataset import make_grpo_data_module
except ImportError:
    make_dpo_data_module = None   # ujson not installed; only needed for DPO/GRPO
    make_grpo_data_module = None

__all__ = [
    "make_dpo_data_module",
    "make_supervised_data_module",
    "make_grpo_data_module",
    "make_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr_fixedToken",
    "make_packed_supervised_data_module_dimv",
    "make_packed_supervised_data_module_dimv_roi",
]