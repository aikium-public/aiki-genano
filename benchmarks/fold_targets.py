"""Fold targets.fasta into PDBs for IgGM consumption."""
import os, time, torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import SeqIO

OUT_DIR = "benchmark_out/esmfold"
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
mdl = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", torch_dtype=torch.float32, low_cpu_mem_usage=True
).cuda().eval()
mdl.esm = mdl.esm.half()          # backbone to fp16
mdl.trunk.set_chunk_size(64)       # smaller chunk -> lower attention peak

def to_pdb_str(out):
    pos = atom14_to_atom37(out["positions"][-1], out).cpu().numpy()
    out = {k: v.to("cpu").numpy() for k, v in out.items()}
    aa = out["aatype"][0]
    return to_pdb(OFProtein(
        aatype=aa, atom_positions=pos[0], atom_mask=out["atom37_atom_exists"][0],
        residue_index=out["residue_index"][0] + 1, b_factors=out["plddt"][0],
        chain_index=out["chain_index"][0] if "chain_index" in out else None,
    ))

for rec in SeqIO.parse("targets.fasta", "fasta"):
    seq = str(rec.seq).upper()
    torch.cuda.reset_peak_memory_stats()
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    t0 = time.time()
    with torch.no_grad():
        out = mdl(ids)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    plddt = out["plddt"].mean().item()
    pdb_path = os.path.join(OUT_DIR, f"{rec.id}.pdb")
    with open(pdb_path, "w") as f:
        f.write(to_pdb_str(out))
    print(f"[esmfold] {rec.id}: {len(seq)} AA  {dt:.1f}s  peak={peak:.2f} GB  plddt={plddt:.2f}  -> {pdb_path}", flush=True)
    torch.cuda.empty_cache()
