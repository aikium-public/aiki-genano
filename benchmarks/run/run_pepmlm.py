"""PepMLM runner: target-conditioned 10-AA peptide binder generation.

Output: benchmark_out/pepmlm/generated_T0.7_seed{seed}.fasta

NOTE: outputs are 10-AA peptides, NOT nanobodies. PepMLM's scaffold mismatch
is the point of interest for the comparison — it's the only fast tool that
does real target-sequence conditioning.
"""
import argparse, os, sys, time, torch, random
from common import load_targets, write_fasta, write_status, already_done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--binder-len", type=int, default=10)
    args = ap.parse_args()

    out_fasta = os.path.join(args.out_dir, f"generated_T{args.T}_seed{args.seed}.fasta")
    os.makedirs(args.out_dir, exist_ok=True)
    targets = load_targets(args.targets)
    total_expected = len(targets) * args.n
    if already_done(out_fasta, total_expected):
        print(f"[pepmlm] already have {total_expected} seqs; skipping"); return

    random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    MODEL = "ChatterjeeLab/PepMLM-650M"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForMaskedLM.from_pretrained(MODEL).cuda().eval()
    print(f"[pepmlm] model ready", flush=True)
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in aa_vocab]).cuda()

    records = []
    t0 = time.time()
    for t_i, target in enumerate(targets):
        masked = target.epitope + (tok.mask_token * args.binder_len)
        enc = tok(masked, return_tensors="pt").to("cuda")
        produced = 0
        while produced < args.n:
            try:
                with torch.no_grad():
                    logits = mdl(**enc).logits[0]
            except Exception as e:
                print(f"[pepmlm] err {target.uniprot_id}: {e}", file=sys.stderr); break
            mask_pos = (enc.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
            sampled = []
            for mi in mask_pos.tolist():
                l = logits[mi, aa_ids] / args.T
                probs = torch.softmax(l, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).item()
                sampled.append(aa_vocab[idx])
            binder = "".join(sampled)
            header = f"pepmlm|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
            records.append((header, binder))
            produced += 1
        dt = time.time() - t0
        print(f"[pepmlm] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced}  ({dt:.0f}s)", flush=True)

    write_fasta(out_fasta, records)
    write_status(args.out_dir, "pepmlm", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "binder_len": args.binder_len,
        "seconds": time.time() - t0, "note": "outputs are peptides, not VHHs",
    })

if __name__ == "__main__":
    main()
