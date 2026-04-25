"""nanoBERT runner: CDR-infill on a fixed VHH template.

Output: benchmark_out/nanobert/generated_T0.7_seed{seed}.fasta

Epitope is noted in header only (nanoBERT has no epitope conditioning API).
For each sample we mask CDR1+CDR2+CDR3 positions on the canonical template
and sample the fills at T from the per-position softmax. Not batched because
the masked positions matter per-sample.
"""
import argparse, os, sys, time, torch, random
from common import (load_targets, write_fasta, write_status, already_done, is_valid_vhh,
                    VHH_TEMPLATE_FILLED, CDR1_POSITIONS, CDR2_POSITIONS, CDR3_POSITIONS)

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
        print(f"[nanobert] already have {total_expected} seqs; skipping")
        return

    random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    MODEL = "NaturalAntibody/nanoBERT"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForMaskedLM.from_pretrained(MODEL).cuda().eval()
    print(f"[nanobert] model ready", flush=True)

    mask_positions = CDR1_POSITIONS + CDR2_POSITIONS + CDR3_POSITIONS
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in aa_vocab]).cuda()

    records = []
    valid_count = 0
    t0 = time.time()

    for t_i, target in enumerate(targets):
        produced = 0
        while produced < args.n:
            residues = list(VHH_TEMPLATE_FILLED)
            for p in mask_positions:
                residues[p] = tok.mask_token
            text = " ".join(residues)
            enc = tok(text, return_tensors="pt").to("cuda")
            try:
                with torch.no_grad():
                    logits = mdl(**enc).logits[0]
            except Exception as e:
                print(f"[nanobert] gen error {target.uniprot_id}: {e}", file=sys.stderr)
                break
            mask_idx = (enc.input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
            filled = list(VHH_TEMPLATE_FILLED)
            for i, mi in enumerate(mask_idx.tolist()):
                l = logits[mi, aa_ids] / args.T
                probs = torch.softmax(l, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1).item()
                filled[mask_positions[i]] = aa_vocab[sampled_idx]
            seq = "".join(filled).upper()
            header = f"nanobert|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
            records.append((header, seq))
            if is_valid_vhh(seq):
                valid_count += 1
            produced += 1
        dt = time.time() - t0
        print(f"[nanobert] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced} seqs  ({dt:.0f}s total, valid={valid_count})", flush=True)

    write_fasta(out_fasta, records)
    write_status(args.out_dir, "nanobert", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "valid_vhh": valid_count,
        "seconds": time.time() - t0,
    })

if __name__ == "__main__":
    main()
