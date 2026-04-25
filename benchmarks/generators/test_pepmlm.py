"""Smoke test for PepMLM (ChatterjeeLab/PepMLM-650M).
Target-conditioned generation: concatenate target + mask tokens at C-terminus,
let the MLM fill the binder region.
NOTE: HF repo has a gated agreement asking for non-commercial research use,
co-authorship on publications, and no patent filings on generated sequences.
"""
import time, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "ChatterjeeLab/PepMLM-650M"
# Target: a short receptor fragment (CD20 extracellular loop, arbitrary demo target)
TARGET  = "DIQMTQSPSSLSASVGDRVTITC"    # test_peptide_02
BINDER_LEN = 10

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
mdl = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()
print(f"[pepmlm] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

masked = TARGET + (tok.mask_token * BINDER_LEN)
enc = tok(masked, return_tensors="pt").to(device)
print(f"[pepmlm] input: {masked[:40]}...  total_ids={enc.input_ids.shape}", flush=True)

for run in range(3):
    with torch.no_grad():
        logits = mdl(**enc).logits[0]
    mask_positions = (enc.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    T = 0.7
    sampled = []
    for mi in mask_positions.tolist():
        l = logits[mi] / T
        # restrict to 20 canonical AA (skip special/ambiguous)
        aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
        aa_ids = [tok.convert_tokens_to_ids(a) for a in aa_vocab]
        lr = torch.tensor([l[i].item() for i in aa_ids])
        probs = torch.softmax(lr, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        sampled.append(aa_vocab[idx])
    binder = "".join(sampled)
    print(f">pepmlm_{run}  target_len={len(TARGET)} binder_len={len(binder)}")
    print(f"binder: {binder}")
