"""Smoke test for nanoBERT (NaturalAntibody/nanoBERT).
Mask a few CDR positions of a canonical VHH template and sample fills.
"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "NaturalAntibody/nanoBERT"

# Canonical camelid VHH template (IgLM-like; 119 AA). Uses '*' to mark positions we will mask.
# CDR1 ~27-37, CDR2 ~57-64, CDR3 ~105-117 (Kabat approx.)
TEMPLATE = "QVQLVESGGGSVQAGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAIDSGGGSTHYADSVKGRFTISRDNAKNTLYLQMNSLKSEDTALYYCAAAAAAAAAAAWGQGTQVTVSS"
# Positions to mask (0-indexed) within CDR1 and CDR3
MASK_POSITIONS = [29, 30, 31, 99, 100, 101, 102, 103]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
mdl = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()
print(f"[nanoBERT] loaded in {time.time()-t0:.1f}s on {device}", flush=True)
print(f"[nanoBERT] mask token: {tok.mask_token!r} id={tok.mask_token_id}", flush=True)

# Build masked sequence as space-separated tokens (BERT-style per-residue tokenizers are typical)
residues = list(TEMPLATE)
for p in MASK_POSITIONS:
    residues[p] = tok.mask_token
text = " ".join(residues)

for run in range(3):
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits[0]
    # Find mask positions in the tokenised input
    mask_idx = (enc.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    # Sample T=0.7
    T = 0.7
    filled = list(residues)
    for i, mi in enumerate(mask_idx.tolist()):
        l = logits[mi] / T
        probs = torch.softmax(l, dim=-1)
        sampled_id = torch.multinomial(probs, num_samples=1).item()
        aa = tok.convert_ids_to_tokens(sampled_id)
        filled[MASK_POSITIONS[i]] = aa
    seq = "".join(filled)
    print(f">nanobert_{run}  len={len(seq)}")
    print(seq)
