"""ESMFold prefolder: fold each target epitope into a PDB for IgGM.

Output: benchmark_out/esmfold_prefolds/<uniprot_id>.pdb (one per target)
"""
import argparse, os, time, torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from common import load_targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    targets = load_targets(args.targets)
    to_fold = [t for t in targets
               if not os.path.exists(os.path.join(args.out_dir, f"{t.uniprot_id}.pdb"))]
    if not to_fold:
        print(f"[esmfold] all {len(targets)} already folded; skipping")
        return

    MODEL = "facebook/esmfold_v1"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = EsmForProteinFolding.from_pretrained(MODEL, torch_dtype=torch.float32,
                                               low_cpu_mem_usage=True).cuda().eval()
    mdl.esm = mdl.esm.half()
    mdl.trunk.set_chunk_size(64)
    print(f"[esmfold] model ready; folding {len(to_fold)}", flush=True)

    def to_pdb_str(out):
        pos = atom14_to_atom37(out["positions"][-1], out).cpu().numpy()
        o = {k: v.to("cpu").numpy() for k, v in out.items()}
        return to_pdb(OFProtein(
            aatype=o["aatype"][0], atom_positions=pos[0],
            atom_mask=o["atom37_atom_exists"][0],
            residue_index=o["residue_index"][0] + 1, b_factors=o["plddt"][0],
            chain_index=o["chain_index"][0] if "chain_index" in o else None,
        ))

    for t in to_fold:
        torch.cuda.reset_peak_memory_stats()
        ids = tok([t.epitope], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        t0 = time.time()
        with torch.no_grad():
            out = mdl(ids)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        plddt = out["plddt"].mean().item()
        pdb_path = os.path.join(args.out_dir, f"{t.uniprot_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(to_pdb_str(out))
        print(f"[esmfold] {t.uniprot_id}: {len(t.epitope)} AA  {dt:.1f}s  peak={peak:.2f}GB  plddt={plddt:.2f}", flush=True)
        torch.cuda.empty_cache()

    print(f"[esmfold] done", flush=True)

if __name__ == "__main__":
    main()
