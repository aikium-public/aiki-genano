"""Smoke test for ProtGPT2 (nferruz/ProtGPT2)."""
import os, time
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "nferruz/ProtGPT2"
EPITOPE = "SLNFLGGLPPL"

t0 = time.time()
print(f"[protgpt2] loading...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
print(f"[protgpt2] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

# ProtGPT2 expects protein sequences with newlines every 60 chars and uses
# <|endoftext|> as separator. A bare "epitope\n" prompt doesn't keep it going.
# Feed the epitope as the start of a putative sequence and let it continue.
prompt = f"<|endoftext|>{EPITOPE}"
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
print(f"[protgpt2] prompt token count: {ids.shape[1]}", flush=True)

t1 = time.time()
with torch.no_grad():
    out = mdl.generate(
        ids, max_new_tokens=60, do_sample=True,
        temperature=0.7, top_p=0.9, num_return_sequences=5,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
    )
print(f"[protgpt2] generated in {time.time()-t1:.1f}s, out shape={tuple(out.shape)}", flush=True)

for i, o in enumerate(out):
    full = tok.decode(o, skip_special_tokens=True)
    # ProtGPT2 tokenization typically has newlines every ~60 chars
    clean = full.replace("\n", "")
    print(f">protgpt2_{i}  len={len(clean)}")
    print(clean[:200])
