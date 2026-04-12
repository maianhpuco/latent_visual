"""
Standalone test: Qwen2.5-VL-3B fine-tuning with various attention backends.
Tests flash_attn, xformers, and sdpa to find fastest option on V100.
"""
import os, time, torch
from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def make_dummy_batch(processor, seq_len=512):
    """Create a minimal dummy batch with one image."""
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What is in this image?"}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    # labels = input_ids for LM loss
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def test_attention(attn_impl):
    print(f"\n{'='*50}")
    print(f"Testing: {attn_impl}")
    print('='*50)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            attn_implementation=attn_impl,
            device_map=DEVICE,
        )
        model.train()
        processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)
        
        batch = make_dummy_batch(processor)
        
        # Warmup
        out = model(**batch)
        loss = out.loss
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
        
        # Time 3 forward+backward passes
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(**batch)
            out.loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        avg = sum(times) / len(times)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ✓ SUCCESS")
        print(f"  Avg step time: {avg:.2f}s")
        print(f"  Peak GPU mem:  {mem:.2f} GB")
        del model
        torch.cuda.empty_cache()
        return True, avg
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        torch.cuda.empty_cache()
        return False, None

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Torch: {torch.__version__}")
    
    # Check xformers
    try:
        import xformers
        print(f"xformers: {xformers.__version__}")
    except ImportError:
        print("xformers: NOT installed")

    # Check flash_attn
    try:
        import flash_attn
        print(f"flash_attn: {flash_attn.__version__}")
    except ImportError:
        print("flash_attn: NOT installed")

    results = {}
    for impl in ["sdpa", "eager"]:
        ok, t = test_attention(impl)
        results[impl] = (ok, t)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for impl, (ok, t) in results.items():
        status = f"{t:.2f}s/step" if ok else "FAILED"
        print(f"  {impl:20s}: {status}")
