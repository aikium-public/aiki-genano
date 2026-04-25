"""Smoke test for NanoAbLLaMA (Lab608/NanoAbLLaMA).
Based on ProLLaMA (7B, LLaMA-2 arch). Try a few plausible prompt formats
and see which one yields VHH-shaped sequences.
"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Lab608/NanoAbLLaMA"
TOKEN = os.environ.get("HF_TOKEN")

# Try 4-bit NF4 to fit beside the ~4 GB already used on the RTX 5000 Ada.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, token=TOKEN, quantization_config=bnb, device_map={"": 0},
)
mdl.eval()
print(f"[nanoabllama] loaded in {time.time()-t0:.1f}s  dtype=4bit-nf4", flush=True)

# ProLLaMA uses prompts like:
#   [Generate by superfamily] Superfamily=<Immunoglobulin superfamily>. Seq=
# NanoAbLLaMA fine-tunes ProLLaMA on nanobodies; germline conditioning is
# mentioned in the paper. We'll try a few candidates and pick the one that
# produces a valid-looking VHH start (QVQLV...).
PROMPTS = [
    "[Generate by germline] Germline=<IGHV3-3*01>. Seq=<",
    "[Generate by superfamily] Superfamily=<Immunoglobulin>. Seq=<",
    "Seq=<QVQLV",
    "<s>QVQLVESGGG",
]

for i, prompt in enumerate(PROMPTS):
    ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            ids, max_new_tokens=180, do_sample=True,
            temperature=0.7, top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\n--- prompt {i}: {prompt!r}")
    print(f"RAW: {text[:300]}")
