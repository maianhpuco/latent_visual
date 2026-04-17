import json
import os
import random
from typing import Optional

import torch
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration

TASK_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
LOCAL_VSTAR_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_configs_dir(configs_dir: Optional[str]) -> str:
    candidates = []
    if configs_dir:
        candidates.append(configs_dir)
    candidates.extend(["configs_simea", "configs_maui", "configs"])

    repo_root = _repo_root()
    for candidate in candidates:
        cfg_path = os.path.join(repo_root, candidate, "data_path.yaml")
        if os.path.isfile(cfg_path):
            return candidate

    raise FileNotFoundError("Could not locate a configs directory with data_path.yaml for V* validation.")


def _load_vstar_records(configs_dir: Optional[str] = None):
    resolved_configs_dir = _resolve_configs_dir(configs_dir)
    cfg_path = os.path.join(_repo_root(), resolved_configs_dir, "data_path.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    vstar_cfg = cfg["evaluation"]["vstar_bench"]
    records = []
    with open(vstar_cfg["test_questions"], "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("text", "").endswith(LOCAL_VSTAR_SUFFIX):
                record["text"] = record["text"][: -len(LOCAL_VSTAR_SUFFIX)]
            records.append(record)

    return records, vstar_cfg["root"], resolved_configs_dir


def _resolve_image(image_value, image_dir: Optional[str]):
    if isinstance(image_value, Image.Image):
        return image_value
    if image_dir is None:
        return image_value
    if isinstance(image_value, list):
        return [os.path.join(image_dir, item) if isinstance(item, str) else item for item in image_value]
    return os.path.join(image_dir, image_value)


def _build_messages(image_input, question: str):
    if not isinstance(image_input, list):
        image_input = [image_input]

    content = [{"type": "image", "image": image} for image in image_input]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _extract_answer(response: str) -> str:
    answer = response.split("<answer>")[-1].split("</answer")[0].strip()
    if " " in answer:
        answer = answer.split(" ")[0]
    if len(answer) > 1:
        answer = answer[0]
    return answer


def _is_correct(response: str, ground_truth: str) -> bool:
    return _extract_answer(response).upper() == ground_truth.upper()


class VStarValidator:
    def __init__(
        self,
        model,
        processor,
        val_indices: list,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        configs_dir: Optional[str] = None,
        max_new_tokens: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.val_indices = val_indices
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.records, self.image_dir, self.configs_dir = _load_vstar_records(configs_dir)

    @staticmethod
    def create_fixed_val_set(
        val_fraction: float = 0.30,
        seed: int = 42,
        save_path: str = "checkpoints_dimv/vstar_val_indices.json",
        configs_dir: Optional[str] = None,
    ) -> list:
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                return json.load(f)

        records, _, resolved_configs_dir = _load_vstar_records(configs_dir)
        total = len(records)
        n_val = max(1, int(total * val_fraction))
        rng = random.Random(seed)
        val_indices = sorted(rng.sample(list(range(total)), n_val))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(val_indices, f, indent=2)

        print(
            f"[VStarValidator] Created fixed validation split: {len(val_indices)}/{total} "
            f"samples ({val_fraction * 100:.0f}%) using {resolved_configs_dir}."
        )
        print(f"[VStarValidator] Saved validation indices to {save_path}")
        return val_indices

    def evaluate(self) -> dict:
        was_training = self.model.training
        self.model.eval()

        correct = 0
        per_sample_results = []

        with torch.no_grad():
            for idx in self.val_indices:
                sample_result = self._eval_single(idx)
                per_sample_results.append(sample_result)
                if sample_result["correct"]:
                    correct += 1

        if was_training:
            self.model.train()

        total = len(per_sample_results)
        accuracy = correct / total if total else 0.0
        return {
            "accuracy": round(accuracy, 4),
            "n_correct": correct,
            "n_samples": total,
            "per_sample": per_sample_results,
        }

    def _run_inference_dimv(self, image_input, question: str) -> str:
        if getattr(self.model, "latent_reasoning", None) is None:
            raise RuntimeError("VStarValidator requires a DIMV model with latent_reasoning enabled.")

        messages = _build_messages(image_input, question)
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = Qwen2_5_VLForConditionalGeneration.generate(
            self.model,
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0]

    def _eval_single(self, index: int) -> dict:
        record = self.records[index]
        image = _resolve_image(record["image"], self.image_dir)
        question = record.get("query", record.get("text", "")) + TASK_INSTRUCTION
        prediction = self._run_inference_dimv(image, question)

        return {
            "idx": index,
            "question_id": record.get("question_id", index),
            "question": record.get("query", record.get("text", "")),
            "prediction": prediction,
            "label": record["label"],
            "category": record.get("category", "all"),
            "correct": _is_correct(prediction, record["label"]),
        }
