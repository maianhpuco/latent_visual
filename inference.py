"""
Inference script for Latent Visual Reasoning (LVR) models.

Works for both Stage-1 (SFT) and Stage-2 (RL/GRPO) checkpoints,
including DIMV-style latent reasoning checkpoints.

Usage examples:
  # Single image + question (coconut-LVR checkpoint)
  python inference.py --checkpoint vincentleebang/LVR-7B --image path/to/img.jpg --question "What color is the car?"

  # Single image + question (DIMV checkpoint)
  python inference.py --checkpoint checkpoints_dimv/stage1_3b_dimv_full/checkpoint-1450 \\
      --image path/to/img.jpg --question "What color is the car?" --num_refinement_steps 8

  # Multiple images (e.g. BLINK-style)
  python inference.py --checkpoint vincentleebang/LVR-7B --image img1.jpg img2.jpg --question "Which image is older?"

  # Benchmark evaluation (coconut-LVR)
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark vstar
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark blink
  python inference.py --checkpoint vincentleebang/LVR-7B --benchmark mmvp

  # Benchmark evaluation (DIMV) — sweeps num_refinement_steps automatically
  python inference.py --checkpoint checkpoints_dimv/stage1_3b_dimv_full/checkpoint-1450 --benchmark vstar
  python inference.py --checkpoint checkpoints_dimv/stage1_3b_dimv_full/checkpoint-1450 \\
      --benchmark vstar --dimv_refinement_steps_list 4 8 16 32 64

Options:
  --checkpoint              HuggingFace repo ID or local path to the model checkpoint
  --image                   Path(s) to image file(s) — can be local path or URL
  --question                Question string (required for single inference)
  --lvr_steps               Coconut-LVR latent decoding steps (default: 8)
  --num_refinement_steps    DIMV cross-attn refinement iterations L (default: 8)
  --decoding_strategy       "steps" (default) or "latent" — coconut-LVR only
  --benchmark               Run full benchmark: vstar | blink | mmvp
  --dimv_refinement_steps_list  DIMV L values to sweep (default: 4 8 16 32 64)
  --lvr_steps_list          Coconut-LVR steps to sweep (default: 4 8 16)
  --output_dir              Where to save benchmark results (default: ./eval_results)
  --max_new_tokens          Max tokens to generate (default: 512)
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
    Works for Stage-1 (SFT), Stage-2 (RL), LoRA, and DIMV checkpoints.

    Auto-detection order:
      1. LoRA: adapter_config.json present → merge adapter into base model
      2. DIMV: config.latent_reasoning_config is set → apply DIMV forward patch
                and recover slot_token_ids from the saved tokenizer
      3. Coconut-LVR: standard inference forward patch

    LoRA checkpoints are auto-detected by the presence of adapter_config.json.
    The base model is loaded from base_model_name_or_path in the adapter config,
    then the adapter is loaded and merged for inference.
    """
    import json as _json

    # --- Detect LoRA checkpoint ---
    adapter_cfg_path = os.path.join(checkpoint, "adapter_config.json")
    is_lora = os.path.isfile(adapter_cfg_path)

    if is_lora:
        with open(adapter_cfg_path) as _f:
            _adapter_cfg = _json.load(_f)
        base_model_id = _adapter_cfg["base_model_name_or_path"]
        print(f"Detected LoRA checkpoint. Base model: {base_model_id}")
        print(f"Adapter: {checkpoint}")
        config_source = base_model_id
    else:
        print(f"Loading model from: {checkpoint}")
        config_source = checkpoint

    config = AutoConfig.from_pretrained(config_source)

    # --- Detect DIMV checkpoint ---
    # DIMV models have latent_reasoning_config stored in config.json at training time.
    is_dimv = (
        hasattr(config, "latent_reasoning_config")
        and config.latent_reasoning_config is not None
    )

    if is_dimv:
        print("[DIMV] Detected DIMV latent reasoning checkpoint.")
        print("[DIMV] Applying DIMV forward patch (bottleneck attention mask + slot injection).")
        replace_qwen2_5_with_mixed_modality_forward_lvr(dimv_mode=True)
    else:
        # Coconut-LVR: standard inference-mode forward
        replace_qwen2_5_with_mixed_modality_forward_lvr(
            inference_mode=True,
            lvr_head=getattr(config, "lvr_head", False),
        )

    # Fix numeric stability in the 3D-conv patch embedding (safe to keep at inference)
    replace_qwen_2_5_vl_patch_emb()

    # DIMV must use sdpa — NOT flash_attention_2.
    #
    # The DIMV forward passes a 4D float mask [B, 1, seq_len, seq_len] with
    # 0.0 (allowed) and -inf (blocked) values to enforce the bottleneck.
    # sdpa adds this mask to attention logits correctly.
    # flash_attention_2 routes any non-None mask through _upad_input() which
    # expects a 2D boolean mask, producing garbage cu_seqlens and CUDA OOB.
    #
    # DIMV was trained with --disable_flash_attn2 True (sdpa); inference must
    # match to avoid the cu_seqlens_q shape error and IndexKernel OOB crashes.
    if is_dimv:
        attn_impl = "sdpa"
        print("[DIMV] Using sdpa attention (flash_attn2 incompatible with 4D bottleneck mask).")
    else:
        # Coconut-LVR: use flash_attention_2 if available
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

    if is_lora:
        from peft import PeftModel
        base_model = QwenWithLVR.from_pretrained(
            base_model_id,
            config=config,
            torch_dtype="auto",
            attn_implementation=attn_impl,
            device_map=device_map,
            max_memory=max_memory,
        )
        print("Loading LoRA adapter and merging weights...")
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully.")
    else:
        model = QwenWithLVR.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype="auto",           # respects bfloat16 stored in config
            attn_implementation=attn_impl,
            device_map=device_map,
            max_memory=max_memory,
        )

    model.eval()

    # DIMV: device_map="auto" leaves LatentReasoningModule on the meta/cpu device
    # because accelerate's dispatch plan only covers standard transformer layers.
    # Fix: load its weights directly from the checkpoint safetensors and materialize
    # on cuda:0 using to_empty() (cannot use .to() on meta tensors).
    if is_dimv and hasattr(model, "latent_reasoning") and model.latent_reasoning is not None:
        _bad_params = [p for p in model.latent_reasoning.parameters()
                       if p.device.type in ("meta", "cpu")]
        if _bad_params:
            import glob
            import safetensors.torch as _st

            # Get dtype from layers that ARE on GPU
            _dtype = next(
                (p.dtype for p in model.parameters() if p.device.type == "cuda"),
                torch.bfloat16,
            )

            # Read latent_reasoning.* tensors from checkpoint shards
            lr_state = {}
            for shard_f in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
                for k, v in _st.load_file(shard_f, device="cpu").items():
                    if k.startswith("latent_reasoning."):
                        lr_state[k[len("latent_reasoning."):]] = v.to(dtype=_dtype)

            if lr_state:
                # Remove accelerate CPU-offload hooks so .to_empty() / load_state_dict work
                try:
                    from accelerate.hooks import remove_hook_from_submodules
                    remove_hook_from_submodules(model.latent_reasoning)
                except Exception:
                    pass
                # Materialize empty tensors on cuda:0, then fill with checkpoint weights
                model.latent_reasoning.to_empty(device="cuda:0")
                model.latent_reasoning.load_state_dict(lr_state, strict=True)
                print(
                    f"[DIMV] Materialized latent_reasoning on cuda:0 "
                    f"(dtype={_dtype}, {len(lr_state)} tensors)"
                )
            else:
                print(
                    "[DIMV] WARNING: latent_reasoning.* not found in checkpoint safetensors — "
                    "Z injection will likely fail."
                )

    # LVR tokens (<|lvr_start|>, <|lvr|>, etc.) are in the checkpoint tokenizer.
    # For LoRA checkpoints the tokenizer is saved alongside the adapter.
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        min_pixels=128 * 28 * 28,    # matches training MIN_TOKEN=128
        max_pixels=2560 * 28 * 28,   # matches training MAX_TOKEN=2560
    )

    tok = processor.tokenizer

    if is_dimv and model.latent_reasoning is not None:
        # DIMV: recover [SLOT_0]..[SLOT_{T_v-1}] token IDs from the saved tokenizer.
        # These were added by setup_slot_tokens() during training.
        T_v = model.T_v
        slot_ids = [tok.convert_tokens_to_ids(f"[SLOT_{k}]") for k in range(T_v)]
        unk_id = tok.unk_token_id
        if all(sid != unk_id for sid in slot_ids):
            model.slot_token_ids = slot_ids
            print(f"[DIMV] Recovered {T_v} slot token IDs: {slot_ids[0]}...{slot_ids[-1]}")
        else:
            print(
                "[DIMV] WARNING: slot tokens [SLOT_k] not found in tokenizer. "
                "Z injection will be skipped — check the checkpoint tokenizer."
            )
    else:
        # Coconut-LVR: patch standard LVR token IDs onto config if missing
        # (happens when base model config is used for LoRA — IDs live in the checkpoint tokenizer)
        _lvr_attrs = {
            "lvr_id":            "<|lvr|>",
            "lvr_start_id":      "<|lvr_start|>",
            "lvr_end_id":        "<|lvr_end|>",
            "lvr_latent_end_id": "<|lvr_latent_end|>",
        }
        for attr, token in _lvr_attrs.items():
            if not hasattr(model.config, attr):
                token_id = tok.convert_tokens_to_ids(token)
                if token_id != tok.unk_token_id:
                    setattr(model.config, attr, token_id)

    print("Model loaded successfully.")
    print(f"  is_dimv={is_dimv}  |  is_lora={is_lora}")
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


def run_inference_dimv(
    model,
    processor,
    image_input,
    question: str,
    num_refinement_steps: int = 2,
    max_new_tokens: int = 512,
) -> str:
    """
    Run a single DIMV inference pass and return the decoded output string.

    For DIMV, reasoning happens *during prefill*: the LatentReasoningModule
    runs `num_refinement_steps` (L) cross-attention iterations to compute the
    slot embeddings Z from the observed context X_o = concat(V, x_txt).
    After prefill, standard autoregressive decoding generates Y.

    Varying L at inference time is valid test-time compute scaling:
    - The cross-attention weights are *shared* across all L iterations.
    - The model was trained with L=2; running more iterations refines Z
      against X_o further, potentially extracting richer visual information.
    - Suggested sweep: L ∈ {4, 8, 16, 32, 64}.

    Args:
        model:                 QwenWithLVR instance loaded with DIMV forward.
        processor:             AutoProcessor with slot tokens in its tokenizer.
        image_input:           Path/URL/PIL image (or list for multi-image).
        question:              Question string.
        num_refinement_steps:  L — cross-attn iterations in LatentReasoningModule.
        max_new_tokens:        Maximum answer tokens to generate.

    Returns:
        Decoded output string (including special tokens, for answer extraction).
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    assert model.latent_reasoning is not None, (
        "run_inference_dimv called on a non-DIMV model "
        "(model.latent_reasoning is None). Use run_inference() instead."
    )

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

    # Temporarily override L for this call and restore afterwards.
    # The LatentReasoningModule's cross-attention weights are shared across
    # all L steps, so changing L at inference is safe.
    orig_L = model.latent_reasoning.L
    model.latent_reasoning.L = num_refinement_steps
    try:
        with torch.no_grad():
            # Bypass QwenWithLVR.generate() (which routes to coconut-LVR
            # decoding machinery) and call the parent's standard generation.
            # The DIMV forward is already patched at the
            # Qwen2_5_VLForConditionalGeneration.forward level, so the parent
            # generate will correctly invoke Z-slot injection during prefill
            # and standard KV-cache decoding afterwards.
            generated_ids = Qwen2_5_VLForConditionalGeneration.generate(
                model,
                **inputs,
                max_new_tokens=max_new_tokens,
            )
    finally:
        model.latent_reasoning.L = orig_L

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
    return extract_answer(response).upper() == ground_truth.upper()


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_vstar(data_root: str, configs_dir: str = "configs"):
    """
    Load V* bench from local JSONL files (uses configs/data_path.yaml paths).
    Falls back to HuggingFace datasets if local files not found.
    """
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), configs_dir, "data_path.yaml")
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


def load_mmvp(data_root: str, configs_dir: str = "configs"):
    """
    Load MMVP benchmark from local CSV + image directory.

    MMVP structure:
      - 300 questions, 150 paired question groups (1&2, 3&4, …).
      - Each pair tests the same image with the same question but opposite
        correct answers (one expects A, the other B).
      - Standard MMVP metric = pair accuracy: % of pairs where BOTH are right.
      - Images: {Index}.jpg  (1.jpg … 300.jpg) in mmvp_cfg["images_dir"].

    Returns:
        records  — list of dicts with keys:
                   question_id, pair_id, image (filename), query, label (A/B)
        image_dir — absolute path to the images folder
    """
    import yaml, re
    cfg_path = os.path.join(os.path.dirname(__file__), configs_dir, "data_path.yaml")
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

            # Parse options: "(a) Open (b) Closed" → two clean lines
            opts_raw = row["Options"].strip()
            # Split on "(b)" to separate option A text from option B text
            parts = re.split(r'\(b\)', opts_raw, maxsplit=1)
            opt_a = re.sub(r'^\(a\)\s*', '', parts[0]).strip()
            opt_b = parts[1].strip() if len(parts) > 1 else ""
            options_str = f"A. {opt_a}\nB. {opt_b}"

            query = row["Question"].strip() + "\nOptions:\n" + options_str

            # Normalize label: "(a)" → "A", "(b)" → "B"
            label_raw = row["Correct Answer"].strip().lower()
            if "(a)" in label_raw or label_raw == "a":
                label = "A"
            elif "(b)" in label_raw or label_raw == "b":
                label = "B"
            else:
                label = label_raw.upper()

            # pair_id: questions 1&2 → pair 1, 3&4 → pair 2, …
            pair_id = (idx - 1) // 2 + 1

            records.append({
                "question_id": idx,
                "pair_id": pair_id,
                "image": f"{idx}.jpg",
                "query": query,
                "label": label,
            })

    print(f"[MMVP] Loaded {len(records)} questions ({len(records)//2} pairs) from {csv_path}")
    return records, image_dir


def compute_mmvp_pair_accuracy(results: list) -> dict:
    """
    Compute MMVP pair-level accuracy from per-question results.

    MMVP standard metric: a pair is "correct" only when BOTH questions in the
    pair are answered correctly. Pair accuracy = correct_pairs / total_pairs.

    Args:
        results: list of dicts, each with 'pair_id', 'prediction', 'label'.

    Returns:
        dict with 'pair_accuracy', 'correct_pairs', 'total_pairs',
        and 'per_question_accuracy' for reference.
    """
    from collections import defaultdict
    pairs = defaultdict(list)
    for r in results:
        pairs[r["pair_id"]].append(is_correct(r["prediction"], r["label"]))

    total_pairs = len(pairs)
    correct_pairs = sum(1 for correct_list in pairs.values() if all(correct_list))
    total_q = sum(len(v) for v in pairs.values())
    correct_q = sum(sum(v) for v in pairs.values())

    return {
        "pair_accuracy":      correct_pairs / total_pairs * 100 if total_pairs else 0,
        "correct_pairs":      correct_pairs,
        "total_pairs":        total_pairs,
        "per_question_accuracy": correct_q / total_q * 100 if total_q else 0,
        "correct_questions":  correct_q,
        "total_questions":    total_q,
    }


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

# Coconut-LVR: number of latent decoding tokens to sweep
STEP_LIST = [4, 8, 16]

# DIMV: cross-attention refinement iterations (L) to sweep at inference time.
# Trained with L=2. Always include L=2 as the training-time baseline.
# Higher L = more test-time compute (shared weights applied more times).
# Expected: L=2 is the anchor; L=4/8 may help if 2 steps under-converged;
# L=16+ risks distribution shift (Z^L outside the LLM's training distribution).
DIMV_REFINEMENT_STEPS = [2, 4, 8, 16, 32, 64]


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
                if "pair_id" in dat:
                    r["pair_id"] = dat["pair_id"]
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

        # MMVP pair accuracy (standard metric)
        if bench_name == "mmvp" and any("pair_id" in r for r in results):
            mmvp_stats = compute_mmvp_pair_accuracy(results)
            print(
                f"  [MMVP pair] {mmvp_stats['correct_pairs']}/{mmvp_stats['total_pairs']} pairs = "
                f"{mmvp_stats['pair_accuracy']:.2f}%  "
                f"(per-question: {mmvp_stats['per_question_accuracy']:.2f}%)"
            )
            results_by_steps[steps] = {
                "total": total, "correct": correct,
                "mmvp_pair_accuracy": mmvp_stats["pair_accuracy"],
                "mmvp_correct_pairs": mmvp_stats["correct_pairs"],
                "mmvp_total_pairs": mmvp_stats["total_pairs"],
            }
        else:
            results_by_steps[steps] = {"total": total, "correct": correct}

    is_mmvp_sweep = bench_name == "mmvp" and any(
        "mmvp_pair_accuracy" in v for v in results_by_steps.values()
    )
    print("\nSummary (overall accuracy by steps):")
    if is_mmvp_sweep:
        print(f"{'steps':>6}  {'per-Q':>8}  {'pair%':>8}  {'pairs':>14}")
        for s, v in results_by_steps.items():
            q_acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            pair_acc = v.get("mmvp_pair_accuracy", float("nan"))
            pairs_str = f"{v.get('mmvp_correct_pairs','?')}/{v.get('mmvp_total_pairs','?')}"
            print(f"{s:>6}  {q_acc:>7.2f}%  {pair_acc:>7.2f}%  {pairs_str:>14}")
    else:
        print(",".join([f"steps={s}: {v['correct']/v['total']*100:.2f}%" for s, v in results_by_steps.items()]))


def run_benchmark_dimv(
    model, processor, bench_name: str,
    records, image_dir,
    output_dir: str,
    dimv_refinement_steps_list: list,
    max_new_tokens: int,
):
    """
    Run a benchmark sweeping over DIMV num_refinement_steps values.

    For each L in dimv_refinement_steps_list:
      - Temporarily sets model.latent_reasoning.L = L
      - Runs inference on all benchmark records
      - Saves per-sample predictions to dimv_L{L:03d}.json
      - Reports per-category and overall accuracy

    Results are cached: if dimv_L{L:03d}.json already exists the run is skipped.
    A final table compares accuracy across all L values.

    Why sweep L?
      The LatentReasoningModule uses *shared* cross-attention weights for all L
      iterations, so changing L at inference is a clean test-time compute sweep:
        L=2  (training default) → baseline
        L=4,8,16,32,64         → more refinement of Z slots against X_o
    """
    os.makedirs(output_dir, exist_ok=True)
    results_by_steps = {}

    for L in dimv_refinement_steps_list:
        out_file = os.path.join(output_dir, f"dimv_L{L:03d}.json")
        total, correct = 0, 0
        by_category = {}

        if os.path.exists(out_file):
            print(f"[{bench_name}] dimv_L={L}: loading cached results from {out_file}")
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
            for dat in tqdm(records, desc=f"[{bench_name}] dimv_L={L}"):
                img = _resolve_image(dat["image"], image_dir)
                question = dat.get("query", dat.get("text", "")) + TASK_INSTRUCTION
                pred = run_inference_dimv(
                    model, processor, img, question,
                    num_refinement_steps=L,
                    max_new_tokens=max_new_tokens,
                )
                cat = dat.get("category", "all")
                by_category.setdefault(cat, {"total": 0, "correct": 0})
                r = {
                    "question_id": dat["question_id"],
                    "prediction": pred,
                    "label": dat["label"],
                    "category": cat,
                    "dimv_L": L,
                }
                if "pair_id" in dat:
                    r["pair_id"] = dat["pair_id"]
                results.append(r)
                if is_correct(pred, dat["label"]):
                    correct += 1
                    by_category[cat]["correct"] += 1
                total += 1
                by_category[cat]["total"] += 1
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

        acc = correct / total * 100 if total > 0 else 0
        print(f"[{bench_name}] dimv_L={L:2d}  Accuracy: {correct}/{total} = {acc:.2f}%")
        for cat, v in sorted(by_category.items()):
            cat_acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"  {cat}: {v['correct']}/{v['total']} = {cat_acc:.2f}%")

        # MMVP pair accuracy (standard metric)
        if bench_name == "mmvp" and any("pair_id" in r for r in results):
            mmvp_stats = compute_mmvp_pair_accuracy(results)
            print(
                f"  [MMVP pair] {mmvp_stats['correct_pairs']}/{mmvp_stats['total_pairs']} pairs = "
                f"{mmvp_stats['pair_accuracy']:.2f}%  "
                f"(per-question: {mmvp_stats['per_question_accuracy']:.2f}%)"
            )
            results_by_steps[L] = {
                "total": total, "correct": correct,
                "mmvp_pair_accuracy": mmvp_stats["pair_accuracy"],
                "mmvp_correct_pairs": mmvp_stats["correct_pairs"],
                "mmvp_total_pairs": mmvp_stats["total_pairs"],
            }
        else:
            results_by_steps[L] = {"total": total, "correct": correct}

    # Summary table
    is_mmvp_sweep = bench_name == "mmvp" and any(
        "mmvp_pair_accuracy" in v for v in results_by_steps.values()
    )
    print(f"\n{'='*60}")
    print(f"DIMV refinement step sweep — {bench_name} accuracy summary")
    if is_mmvp_sweep:
        print(f"{'L':>6}  {'per-Q':>8}  {'pair%':>8}  {'pairs':>14}")
        print(f"{'-'*45}")
        for L, v in results_by_steps.items():
            q_acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            pair_acc = v.get("mmvp_pair_accuracy", float("nan"))
            pairs_str = f"{v.get('mmvp_correct_pairs','?')}/{v.get('mmvp_total_pairs','?')}"
            print(f"{L:>6}  {q_acc:>7.2f}%  {pair_acc:>7.2f}%  {pairs_str:>14}")
    else:
        print(f"{'L':>6}  {'correct':>8}  {'total':>8}  {'accuracy':>10}")
        print(f"{'-'*40}")
        for L, v in results_by_steps.items():
            acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
            print(f"{L:>6}  {v['correct']:>8}  {v['total']:>8}  {acc:>9.2f}%")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LVR / DIMV Inference")
    parser.add_argument("--checkpoint", required=True,
                        help="HuggingFace repo ID or local path to model checkpoint")
    parser.add_argument("--image", nargs="+", default=None,
                        help="Image path(s) or URL(s) for single inference")
    parser.add_argument("--question", default=None,
                        help="Question string for single inference")

    # --- Coconut-LVR decoding ---
    parser.add_argument("--lvr_steps", type=int, default=8,
                        help="Coconut-LVR latent decoding steps for single inference (default: 8)")
    parser.add_argument("--decoding_strategy", default="steps",
                        choices=["steps", "latent"],
                        help="Coconut-LVR decoding strategy (default: steps)")
    parser.add_argument("--lvr_steps_list", nargs="+", type=int, default=STEP_LIST,
                        help="Coconut-LVR steps to sweep in benchmark (default: 4 8 16)")

    # --- DIMV cross-attention refinement ---
    parser.add_argument("--num_refinement_steps", type=int, default=8,
                        help=(
                            "DIMV: number of cross-attention refinement iterations L "
                            "for single inference (default: 8). "
                            "Model was trained with L=2; higher values = more test-time compute."
                        ))
    parser.add_argument("--dimv_refinement_steps_list", nargs="+", type=int,
                        default=DIMV_REFINEMENT_STEPS,
                        help=(
                            "DIMV: L values to sweep when running benchmark "
                            f"(default: {DIMV_REFINEMENT_STEPS})"
                        ))

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--benchmark", default=None,
                        choices=["vstar", "blink", "mmvp"],
                        help="Run a full benchmark evaluation")
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--configs_dir", default="configs",
                        help="Directory containing data_path.yaml (default: configs; use configs_simea on simea)")
    return parser.parse_args()


def main():
    args = parse_args()

    model, processor = load_model_and_processor(args.checkpoint)

    # Detect which mode we're in
    is_dimv = model.latent_reasoning is not None

    # ---- Single inference ----
    if args.benchmark is None:
        if args.image is None or args.question is None:
            print("ERROR: --image and --question are required for single inference (or use --benchmark).")
            sys.exit(1)
        images = args.image if len(args.image) > 1 else args.image[0]

        if is_dimv:
            print(f"[DIMV] Single inference with L={args.num_refinement_steps} refinement steps.")
            output = run_inference_dimv(
                model, processor,
                image_input=images,
                question=args.question,
                num_refinement_steps=args.num_refinement_steps,
                max_new_tokens=args.max_new_tokens,
            )
        else:
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
        records, image_dir = load_vstar(None, configs_dir=args.configs_dir)
    elif bench == "mmvp":
        records, image_dir = load_mmvp(None, configs_dir=args.configs_dir)
    elif bench == "blink":
        records, image_dir = load_blink(None)
    else:
        raise ValueError(f"Unknown benchmark: {bench}")

    if is_dimv:
        print(
            f"[DIMV] Sweeping refinement steps L ∈ {args.dimv_refinement_steps_list}\n"
            f"       (model trained with L=2; higher L = more test-time compute)"
        )
        run_benchmark_dimv(
            model, processor,
            bench_name=bench,
            records=records,
            image_dir=image_dir,
            output_dir=out_dir,
            dimv_refinement_steps_list=args.dimv_refinement_steps_list,
            max_new_tokens=args.max_new_tokens,
        )
    else:
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
