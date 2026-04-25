"""Smoke test for IgLM (Graylab). Camelid heavy-chain generation."""
import time
print("[iglm] importing...", flush=True)
from iglm import IgLM
t0 = time.time()
iglm = IgLM()
print(f"[iglm] loaded in {time.time()-t0:.1f}s", flush=True)

# Unconditional full-chain generation: [HEAVY]/[CAMEL]
seqs = []
t1 = time.time()
for i in range(5):
    s = iglm.generate(chain_token="[HEAVY]", species_token="[CAMEL]",
                      num_to_generate=1, temperature=0.7)
    seqs.append(s[0] if isinstance(s, list) else s)
print(f"[iglm] 5 seqs in {time.time()-t1:.1f}s", flush=True)

for i, s in enumerate(seqs):
    print(f">iglm_{i}  len={len(s)}")
    print(s)
