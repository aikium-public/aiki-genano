"""IgGM runner: structural epitope-conditioned nanobody design.

Output: benchmark_out/iggm/<uniprot>_<idx>.pdb  (all designed complexes)
        benchmark_out/iggm/generated_T{default}_seed{seed}.fasta (sequences)

Requires: benchmark_out/esmfold_prefolds/<uniprot>.pdb for each target.
Runs from the iggm conda env + canonical IgGM install location.

Invokes IgGM's design.py once per target with --num_samples N. Then extracts
the Predicted Sequence for chain H from each output PDB header.
"""
import argparse, os, sys, time, subprocess, shutil, glob, tempfile
from common import load_targets, write_fasta, write_status, already_done, VHH_TEMPLATE_ALL_X

IGGM_ROOT = "${IGGM_REPO:-/opt/IgGM}"
IGGM_PY = "${HOME}/miniforge3/envs/iggm/bin/python"

def build_fasta(epitope: str, out_path: str):
    with open(out_path, "w") as f:
        f.write(">H\n")
        f.write(VHH_TEMPLATE_ALL_X + "\n")
        f.write(">A\n")
        f.write(epitope + "\n")

def extract_designed_sequence(pdb_path: str) -> str | None:
    with open(pdb_path) as f:
        for line in f:
            if "Predicted Sequence for chain H" in line:
                return line.split(":", 1)[1].strip().upper()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefold-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out_fasta = os.path.join(args.out_dir, f"generated_Tdefault_seed{args.seed}.fasta")
    targets = load_targets(args.targets)
    total_expected = len(targets) * args.n
    if already_done(out_fasta, total_expected):
        print(f"[iggm] already have {total_expected}; skipping"); return

    per_target_dir = os.path.join(args.out_dir, "per_target")
    os.makedirs(per_target_dir, exist_ok=True)
    fasta_inputs_dir = os.path.join(args.out_dir, "fasta_inputs")
    os.makedirs(fasta_inputs_dir, exist_ok=True)

    t_global = time.time()
    all_records = []

    for t_i, target in enumerate(targets):
        tgt_out = os.path.join(per_target_dir, target.uniprot_id)
        os.makedirs(tgt_out, exist_ok=True)
        existing = glob.glob(os.path.join(tgt_out, "*_*.pdb"))
        if len(existing) >= args.n:
            print(f"[iggm] {target.uniprot_id}: {len(existing)} already, skipping run", flush=True)
        else:
            # Build input fasta
            fasta_path = os.path.join(fasta_inputs_dir, f"{target.uniprot_id}.fasta")
            build_fasta(target.epitope, fasta_path)
            antigen_pdb = os.path.join(args.prefold_dir, f"{target.uniprot_id}.pdb")
            if not os.path.exists(antigen_pdb):
                print(f"[iggm] {target.uniprot_id}: no prefold at {antigen_pdb}; skip", file=sys.stderr)
                continue
            epitope_resids = list(range(1, len(target.epitope) + 1))
            cmd = [
                IGGM_PY, "design.py",
                "--fasta", fasta_path,
                "--antigen", antigen_pdb,
                "--epitope", *[str(r) for r in epitope_resids],
                "--num_samples", str(args.n),
                "--output", tgt_out,
            ]
            t0 = time.time()
            print(f"[iggm] {t_i+1}/{len(targets)} {target.uniprot_id}: starting {args.n} samples", flush=True)
            env = os.environ.copy()
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            try:
                proc = subprocess.run(cmd, cwd=IGGM_ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True)
                log_path = os.path.join(tgt_out, "iggm_run.log")
                with open(log_path, "w") as lf:
                    lf.write(proc.stdout)
                print(f"[iggm] {target.uniprot_id}: IgGM exit {proc.returncode}, {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"[iggm] {target.uniprot_id}: subprocess err: {e}", file=sys.stderr)
                continue

        # Collect sequences from produced PDBs
        pdbs = sorted(glob.glob(os.path.join(tgt_out, "*_*.pdb")))
        for idx, p in enumerate(pdbs):
            seq = extract_designed_sequence(p)
            if seq:
                header = f"iggm|{target.uniprot_id}|{target.name.replace(' ','_')}|{idx}"
                all_records.append((header, seq))
        print(f"[iggm] {target.uniprot_id}: collected {len(pdbs)} seqs; cumulative {len(all_records)}", flush=True)

    write_fasta(out_fasta, all_records)
    write_status(args.out_dir, "iggm", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(all_records),
        "seconds": time.time() - t_global,
    })

if __name__ == "__main__":
    main()
