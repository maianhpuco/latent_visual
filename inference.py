"""
Inference script for Latent Visual Reasoning (LVR) models.

Works for both Stage-1 (SFT) and Stage-2 (RL/GRPO) checkpoints.

Usage examples:
  # Single image + question
  python inference.py --checkpoint vincentleebang/LVR-7B --image path/to/img.jpg --question "What color is the car?"

  # Multiple images (e.g. BLINK-style)
  python inference.py --checkpoint vincentleebang/LVR-7B --image img1.jpg img2.jpg --question "Which image is older?"

  # Benchmark evaluation
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark vstar
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark blink
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark mmvp

Options:
  --checkpoint   HuggingFace repo ID or local path to the model checkpoint
  --image        Path(s) to image file(s) — can be local path or URL
  --question     Question string (required for single inference)
  --lvr_steps    Number of latent reasoning steps (default: 8)
  --decoding_strategy  "steps" (default) or "latent"
  --benchmark    Run full benchmark: vstar | blink | mmvp
  --output_dir   Where to save benchmark results (default: ./eval_results)
  --max_new_tokens  Max tokens to generate (default: 512)
"""

import sys
import os

# Ensure src/ is on the Python path (required for internal imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import csv
import string
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor, AutoConfig

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(checkpoint: str, device_map: str = "auto"):
    """
    Load a QwenWithLVR model and its processor from a local path or HF repo.
    Works identically for Stage-1 (SFT) and Stage-2 (RL) checkpoints.
    """
    print(f"Loading model from: {checkpoint}")
    # model_type is standard qwen2_5_vl — no trust_remote_code needed
    config = AutoConfig.from_pretrained(checkpoint)

    # Patch the forward function into inference mode before loading weights.
    # config.lvr_head=False for the released LVR-7B checkpoint.
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True,
        lvr_head=getattr(config, "lvr_head", False),
    )

    # Fix numeric stability in the 3D-conv patch embedding (safe to keep at inference)
    replace_qwen_2_5_vl_patch_emb()

    # Use flash_attention_2 if available, otherwise fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        flash_attn.flash_attn_interface  # trigger the real import to catch ABI errors
        attn_impl = "flash_attention_2"
    except Exception:
        print("flash_attn not available or ABI mismatch — using sdpa instead")
        attn_impl = "sdpa"

    # When using device_map="auto" across multiple GPUs, cap GPU 0 lower so the
    # vision encoder activations (large at 5120 image tokens) don't OOM on GPU 0.
    import torch
    n_gpus = torch.cuda.device_count()
    if device_map == "auto" and n_gpus > 1:
        max_memory = {0: "20GiB"}
        max_memory.update({i: "30GiB" for i in range(1, n_gpus)})
        max_memory["cpu"] = "100GiB"
    else:
        max_memory = None

    model = QwenWithLVR.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype="auto",           # respects bfloat16 stored in config
        attn_implementation=attn_impl,
        device_map=device_map,
        max_memory=max_memory,
    )
    model.eval()

    # LVR tokens (<|lvr_start|>, <|lvr|>, etc.) are already in the checkpoint
    # tokenizer — no need to add them here (unlike training)
    # Match training resolution: model was trained with min=128, max=5120 tokens
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        min_pixels=128 * 28 * 28,    # 100,352 px — matches training MIN_TOKEN=128
        max_pixels=2560* 28 * 28, 
        # max_pixels=2560 * 28 * 28,   # 4,014,080 px — matches training MAX_TOKEN=5120
    )
    print("Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def build_messages(image_input, question: str) -> list:
    """
    Build the chat-template message list.
    image_input: a single path/URL string, a PIL Image, or a list of those.
    """
    if not isinstance(image_input, list):
        image_input = [image_input]

    content = []
    for img in image_input:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question})

    return [{"role": "user", "content": content}]


def run_inference(
    model,
    processor,
    image_input,
    question: str,
    lvr_steps: int = 8,
    decoding_strategy: str = "steps",
    max_new_tokens: int = 512,
) -> str:
    """
    Run a single forward pass and return the decoded output string.
    """
    messages = build_messages(image_input, question)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            lvr_steps=[lvr_steps],
        )

    # Strip the prompt tokens from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return output_texts[0]


# ---------------------------------------------------------------------------
# Answer extraction + accuracy
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    """Pull the text inside <answer>...</answer>."""
    answer = response.split("<answer>")[-1].split("</answer")[0].strip()
    # Keep only the first word / first character if multi-char (MCQ style)
    if " " in answer:
        answer = answer.split(" ")[0]
    if len(answer) > 1:
        answer = answer[0]
    return answer


def is_correct(response: str, ground_truth: str) -> bool:
    return extract_answer(response) == ground_truth


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_vstar(data_root: str):
    """
    Load V* bench from local JSONL files (uses configs/data_path.yaml paths).
    Falls back to HuggingFace datasets if local files not found.
    """
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "data_path.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    vstar_cfg = cfg["evaluation"]["vstar_bench"]

    jsonl_path = vstar_cfg["test_questions"]
    img_dirs = {
        "direct_attributes": vstar_cfg["direct_attributes_dir"],
        "relative_position": vstar_cfg["relative_position_dir"],
    }

    if not os.path.exists(jsonl_path):
        print("Local V* files not found, loading from HuggingFace…")
        from datasets import load_dataset
        ds = load_dataset("craigwu/vstar_bench")["test"]
        records = []
        for i, row in enumerate(ds):
            cat = "direct_attributes" if i <= 114 else "relative_position"
            records.append({
                "question_id": row.get("question_id", i),
                "image": row["image"],       # PIL Image
                "text": row["text"],
                "label": row["label"],
                "category": cat,
            })
        return records, None   # image_dir=None means images are embedded

    _SUFFIX = "\nAnswer with the option's letter from the given choices directly."
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            # The local JSONL has the task instruction already embedded in `text`.
            # run_benchmark appends TASK_INSTRUCTION, so strip it here to avoid duplicate.
            if rec.get("text", "").endswith(_SUFFIX):
                rec["text"] = rec["text"][: -len(_SUFFIX)]
            records.append(rec)
    # image field in jsonl is e.g. "direct_attributes/sa_4690.jpg"
    # so image_dir = root resolves the full path correctly
    return records, vstar_cfg["root"]


def load_mmvp(data_root: str):
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "data_path.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mmvp_cfg = cfg["evaluation"]["mmvp"]

    csv_path = mmvp_cfg["questions_csv"]
    image_dir = mmvp_cfg["images_dir"]

    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            # Normalize options from (a)/(b) parenthetical format to A./B. to match training format
            query = row["Question"] + "\nOptions:\n" + row["Options"]
            query = query.replace("(a)", "A.").replace("(b)", "B.")
            # Normalize label: (a) → A, (b) → B
            label = row["Correct Answer"]
            if label in ["(a)", "(b)"]:
                label = label.strip().upper()[1]
            records.append({
                "question_id": idx,
                "image": f"{idx}.jpg",
                "query": query,
                "label": label,
            })
    return records, image_dir


def load_blink(data_root: str):
    """Load a subset of BLINK from HuggingFace (PIL images embedded)."""
    from datasets import load_dataset
    configs = ["Counting", "IQ_Test", "Jigsaw", "Relative_Reflectance", "Spatial_Relation"]
    records = []
    for config in configs:
        ds = load_dataset("BLINK-Benchmark/BLINK", config)["val"]
        for dat in ds:
            choices = dat["choices"]
            letters = string.ascii_uppercase
            option_str = "".join(f"{l}. {c}\n" for l, c in zip(letters, choices))
            ans = dat["answer"][1].upper() if len(dat["answer"]) > 1 else dat["answer"][0].upper()
            images = [
                dat[k] for k in ["image_1", "image_2", "image_3", "image_4"]
                if k in dat and dat[k] is not None
            ]
            records.append({
                "question_id": dat["idx"],
                "image": images,
                "query": dat["question"] + "\nOptions:\n" + option_str,
                "label": ans,
                "category": config,
            })
    return records, None   # images are PIL objects


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

TASK_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
STEP_LIST = [4, 8, 16]   # evaluate at multiple latent step counts


def _resolve_image(img, img_dir):
    """Return image path or PIL Image usable by process_vision_info."""
    if isinstance(img, Image.Image):
        return img
    if img_dir is None:
        return img
    if isinstance(img, list):
        return [os.path.join(img_dir, i) if isinstance(i, str) else i for i in img]
    return os.path.join(img_dir, img)


def run_benchmark(
    model, processor, bench_name: str,
    records, image_dir,
    output_dir: str,
    lvr_steps_list: list,
    decoding_strategy: str,
    max_new_tokens: int,
):
    os.makedirs(output_dir, exist_ok=True)
    results_by_steps = {}

    for steps in lvr_steps_list:
        out_file = os.path.join(output_dir, f"{decoding_strategy}{steps:03d}.json")
        total, correct = 0, 0
        by_category = {}

        if os.path.exists(out_file):
            print(f"[{bench_name}] steps={steps}: loading cached results from {out_file}")
            with open(out_file) as f:
                results = json.load(f)
            for r in results:
                cat = r.get("category", "all")
                by_category.setdefault(cat, {"total": 0, "correct": 0})
                if is_correct(r["prediction"], r["label"]):
                    correct += 1
                    by_category[cat]["correct"] += 1
                total += 1
                by_category[cat]["total"] += 1
        else:
            results = []
            for dat in tqdm(records, desc=f"[{bench_name}] steps={steps}"):
                img = _resolve_image(dat["image"], image_dir)
                question = dat.get("query", dat.get("text", "")) + TASK_INSTRUCTION
                pred = run_inference(
                    model, processor, img, question,
                    lvr_steps=steps,
                    decoding_strategy=decoding_strategy,
                    max_new_tokens=max_new_tokens,
                )
                cat = dat.get("category", "all")
                by_category.setdefault(cat, {"total": 0, "correct": 0})
                r = {
                    "question_id": dat["question_id"],
                    "prediction": pred,
                    "label": dat["label"],
                    "category": cat,
                }
                results.append(r)
                if is_correct(pred, dat["label"]):
                    correct += 1
                    by_category[cat]["correct"] += 1
                total += 1
                by_category[cat]["total"] += 1
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

        acc = correct / total * 100 if total > 0 else 0
        print(f"[{bench_name}] steps={steps:2d}  Accuracy: {correct}/{total} = {acc:.2f}%")
        for cat, v in sorted(by_category.items()):
            cat_acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"  {cat}: {v['correct']}/{v['total']} = {cat_acc:.2f}%")
        results_by_steps[steps] = {"total": total, "correct": correct}

    print("\nSummary (overall accuracy by steps):")
    print(",".join([f"steps={s}: {v['correct']/v['total']*100:.2f}%" for s, v in results_by_steps.items()]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LVR Inference")
    parser.add_argument("--checkpoint", required=True,
                        help="HuggingFace repo ID or local path to model checkpoint")
    parser.add_argument("--image", nargs="+", default=None,
                        help="Image path(s) or URL(s) for single inference")
    parser.add_argument("--question", default=None,
                        help="Question string for single inference")
    parser.add_argument("--lvr_steps", type=int, default=8,
                        help="Number of latent reasoning steps (default: 8)")
    parser.add_argument("--decoding_strategy", default="steps",
                        choices=["steps", "latent"],
                        help="Decoding strategy (default: steps)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--benchmark", default=None,
                        choices=["vstar", "blink", "mmvp"],
                        help="Run a full benchmark evaluation")
    parser.add_argument("--lvr_steps_list", nargs="+", type=int, default=STEP_LIST,
                        help="Steps to evaluate when running benchmark (default: 4 8 16)")
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Output directory for benchmark results")
    return parser.parse_args()


def main():
    args = parse_args()

    model, processor = load_model_and_processor(args.checkpoint)

    # ---- Single inference ----
    if args.benchmark is None:
        if args.image is None or args.question is None:
            print("ERROR: --image and --question are required for single inference (or use --benchmark).")
            sys.exit(1)
        images = args.image if len(args.image) > 1 else args.image[0]
        output = run_inference(
            model, processor,
            image_input=images,
            question=args.question,
            lvr_steps=args.lvr_steps,
            decoding_strategy=args.decoding_strategy,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n" + "=" * 60)
        print("Model output:")
        print(output)
        print("Extracted answer:", extract_answer(output))
        return

    # ---- Benchmark evaluation ----
    bench = args.benchmark
    out_dir = os.path.join(args.output_dir, bench)
    print(f"\nRunning benchmark: {bench}  →  results in {out_dir}")

    if bench == "vstar":
        records, image_dir = load_vstar(None)
    elif bench == "mmvp":
        records, image_dir = load_mmvp(None)
    elif bench == "blink":
        records, image_dir = load_blink(None)
    else:
        raise ValueError(f"Unknown benchmark: {bench}")

    run_benchmark(
        model, processor,
        bench_name=bench,
        records=records,
        image_dir=image_dir,
        output_dir=out_dir,
        lvr_steps_list=args.lvr_steps_list,
        decoding_strategy=args.decoding_strategy,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
