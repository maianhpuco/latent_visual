import json, traceback
from transformers import AutoProcessor
from src.dataset.lvr_sft_dataset_packed import IterableSupervisedDatasetLVR

processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', min_pixels=100352, max_pixels=4014080)
meta = json.load(open('/project/hnguyen2/mvu9/datasets/lvr_data/lvr_train/meta_data_lvr_sft_stage1_fixed.json'))[0]

class FakeArgs:
    image_min_pixels = 100352
    image_max_pixels = 4014080
    video_min_pixels = 100352
    video_max_pixels = 4014080
    image_resized_width = None
    image_resized_height = None
    video_resized_width = None
    video_resized_height = None
    fps = None
    random_seed = None

ds = IterableSupervisedDatasetLVR(
    data_path=meta['data_path'],
    image_folder=meta['image_folder'],
    ds_name=meta['ds_name'],
    processor=processor,
    data_args=FakeArgs(),
    model_id='Qwen/Qwen2.5-VL-3B-Instruct',
)

print(f"Dataset length: {len(ds)}")
try:
    sample = next(iter(ds))
    print('SUCCESS, keys:', list(sample.keys()))
    print('input_ids shape:', sample['input_ids'].shape)
except Exception as e:
    traceback.print_exc()
