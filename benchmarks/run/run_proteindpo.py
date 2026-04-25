"""ProteinDPO runner: backbone-conditioned inverse folding.

Output: benchmark_out/proteindpo/generated_T0.7_seed{seed}.fasta

Uses vanilla ESM-IF1 (DPO weights would be a drop-in --weights_path if needed).
Epitope is NOT used — ProteinDPO is backbone-conditioned. Same VHH backbone PDB
is used for every target; the target info is preserved in the header only.
Running on CPU because current fair-esm has a device-handling bug that would
need a monkey-patch for GPU execution.
"""
import argparse, os, sys, time, torch, random, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbone-pdb", required=True,
                    help="Canonical VHH backbone PDB (from IgGM examples)")
    ap.add_argument("--chain", default="H")
    ap.add_argument("--weights", default="${PROTEINDPO_WEIGHTS:-./weights}/esm_if1_gvp4_t16_142M_UR50.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # force CPU for ESM-IF1 sampling (device-handling bug in current fair-esm)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # import AFTER CUDA_VISIBLE_DEVICES is set
    import esm, esm.inverse_folding
    from common import load_targets, write_fasta, write_status, already_done, is_valid_vhh

    out_fasta = os.path.join(args.out_dir, f"generated_T{args.T}_seed{args.seed}.fasta")
    os.makedirs(args.out_dir, exist_ok=True)
    targets = load_targets(args.targets)
    total_expected = len(targets) * args.n
    if already_done(out_fasta, total_expected):
        print(f"[proteindpo] already have {total_expected}; skipping"); return

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.weights)
    model.eval()
    print(f"[proteindpo] model ready (CPU)", flush=True)

    coords, ref_seq = esm.inverse_folding.util.load_coords(args.backbone_pdb, args.chain)
    partial = ['<mask>'] * len(ref_seq)
    print(f"[proteindpo] backbone: {args.backbone_pdb} chain {args.chain}, len={len(ref_seq)}", flush=True)

    records = []
    valid_count = 0
    t0 = time.time()
    for t_i, target in enumerate(targets):
        produced = 0
        while produced < args.n:
            try:
                s = model.sample(coords, partial_seq=partial, temperature=args.T)
            except Exception as e:
                print(f"[proteindpo] err {target.uniprot_id}: {e}", file=sys.stderr)
                break
            s = s.upper().strip()
            header = f"proteindpo|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
            records.append((header, s))
            if is_valid_vhh(s):
                valid_count += 1
            produced += 1
        dt = time.time() - t0
        eta = dt / (t_i + 1) * (len(targets) - t_i - 1)
        # incremental write after each target
        write_fasta(out_fasta, records)
        write_status(args.out_dir, "proteindpo", {
            "n_targets": len(targets), "n_per_target": args.n,
            "targets_done": t_i + 1,
            "total_records": len(records), "valid_vhh": valid_count,
            "backbone_pdb": args.backbone_pdb, "chain": args.chain,
            "seconds": dt, "in_progress": True,
        })
        print(f"[proteindpo] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced}  ({dt:.0f}s, ETA {eta:.0f}s, valid={valid_count})", flush=True)

    write_status(args.out_dir, "proteindpo", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "valid_vhh": valid_count,
        "backbone_pdb": args.backbone_pdb, "chain": args.chain,
        "seconds": time.time() - t0, "in_progress": False,
    })

if __name__ == "__main__":
    main()
