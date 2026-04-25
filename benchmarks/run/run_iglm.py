"""IgLM runner: unconditioned camelid VHH generation.

Output: benchmark_out/iglm/generated_T0.7_seed{seed}.fasta
Header format: >iglm|{uniprot}|{name}|{idx}

Epitope is noted in the header but NOT conditioned on — IgLM has no epitope API.
This is the "scaffold-only camelid VHH baseline".
"""
import argparse, os, sys, time, torch, random
from common import load_targets, write_fasta, write_status, already_done, is_valid_vhh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_fasta = os.path.join(args.out_dir, f"generated_T{args.T}_seed{args.seed}.fasta")
    os.makedirs(args.out_dir, exist_ok=True)

    targets = load_targets(args.targets)
    total_expected = len(targets) * args.n
    if already_done(out_fasta, total_expected):
        print(f"[iglm] already have {total_expected} seqs at {out_fasta}; skipping")
        return

    random.seed(args.seed); torch.manual_seed(args.seed)
    from iglm import IgLM
    mdl = IgLM()
    print(f"[iglm] model ready", flush=True)

    records = []
    valid_count = 0
    t0 = time.time()
    for t_i, target in enumerate(targets):
        produced = 0
        attempts = 0
        max_attempts = args.n * 3
        while produced < args.n and attempts < max_attempts:
            attempts += 1
            try:
                s = mdl.generate(chain_token="[HEAVY]", species_token="[CAMEL]",
                                 num_to_generate=1, temperature=args.T)
                s = s[0] if isinstance(s, list) else s
                s = s.strip().upper()
            except Exception as e:
                print(f"[iglm] gen error on {target.uniprot_id}: {e}", file=sys.stderr)
                continue
            header = f"iglm|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
            records.append((header, s))
            produced += 1
            if is_valid_vhh(s):
                valid_count += 1
        dt = time.time() - t0
        print(f"[iglm] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced} seqs  ({dt:.0f}s total, valid={valid_count})", flush=True)

    write_fasta(out_fasta, records)
    write_status(args.out_dir, "iglm", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "valid_vhh": valid_count,
        "seconds": time.time() - t0,
    })
    print(f"[iglm] done: {len(records)} seqs, {valid_count} valid, {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
