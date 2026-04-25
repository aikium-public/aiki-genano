"""ESMFold smoke test via HuggingFace port (no openfold dependency).
Model: facebook/esmfold_v1 (~3 GB), transformers EsmForProteinFolding.
"""
import os, time, torch
from transformers import AutoTokenizer, EsmForProteinFolding

MODEL = "facebook/esmfold_v1"
SEQ = "SLNFLGGLPPL"  # 11-AA peptide
OUT = "benchmark_out/esmfold/test_peptide_01.pdb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = EsmForProteinFolding.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
mdl = mdl.cuda().eval()
# smaller chunk for tight VRAM; only affects attention
mdl.esm = mdl.esm.half()
mdl.trunk.set_chunk_size(64)
print(f"[esmfold] loaded in {time.time()-t0:.1f}s", flush=True)
print(f"[esmfold] VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

ids = tok([SEQ], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
t1 = time.time()
with torch.no_grad():
    out = mdl(ids)
print(f"[esmfold] folded {SEQ} ({len(SEQ)} AA) in {time.time()-t1:.1f}s", flush=True)
print(f"[esmfold] peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

# Convert output to PDB via transformers' helper
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa, atom_positions=pred_pos, atom_mask=mask,
            residue_index=resid, b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

pdb = convert_outputs_to_pdb(out)[0]
with open(OUT, "w") as f:
    f.write(pdb)
print(f"[esmfold] wrote {OUT} ({len(pdb)} bytes)")
print(f"[esmfold] mean pLDDT: {out['plddt'].mean().item():.1f}")
