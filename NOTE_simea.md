
LORA 3B QWEN 


Checkpoint	Status
checkpoint-500	saved
checkpoint-1000	saved
checkpoint-1500	saved
checkpoint-2000	saved ← best available
 


make eval-vstar-lora
# or with specific steps:
make eval-vstar-lora STEPS_LIST="8"
# or a different checkpoint:
make eval-vstar-lora LORA_CHECKPOINT=checkpoints_rerun/stage1_3b_lora/checkpoint-1500
 