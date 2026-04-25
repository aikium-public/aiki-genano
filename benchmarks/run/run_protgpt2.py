"""ProtGPT2 runner: epitope-prefix prompted continuation.

Output: benchmark_out/protgpt2/generated_T0.7_seed{seed}.fasta

This uses the epitope as a *prefix prompt*. ProtGPT2 has no VHH or epitope
conditioning training, so output will rarely be VHH-shaped. Baseline only.
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
        print(f"[protgpt2] already have {total_expected} seqs; skipping")
        return

    random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    mdl = AutoModelForCausalLM.from_pretrained(
        "nferruz/ProtGPT2", torch_dtype=torch.float16
    ).cuda().eval()
    print(f"[protgpt2] model ready", flush=True)

    records = []
    valid_count = 0
    t0 = time.time()
    # Conservative settings after the earlier run stalled on target 9/10 at
    # batch_size=10 / max_new_tokens=160.
    batch_size = 2
    max_new_tokens = 100
    for t_i, target in enumerate(targets):
        prompt = f"<|endoftext|>{target.epitope}"
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        target_t0 = time.time()
        produced = 0
        stuck_attempts = 0
        while produced < args.n:
            k = min(batch_size, args.n - produced)
            try:
                with torch.no_grad():
                    out = mdl.generate(
                        ids, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=args.T, top_p=0.9,
                        num_return_sequences=k,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
            except Exception as e:
                print(f"[protgpt2] gen error {target.uniprot_id}: {e}", file=sys.stderr)
                break
            # bail out if this target is taking forever (>5 min)
            if time.time() - target_t0 > 300:
                print(f"[protgpt2] {target.uniprot_id}: timeout after 300s at {produced}/{args.n}, moving on",
                      file=sys.stderr, flush=True)
                break
            for o in out:
                txt = tok.decode(o, skip_special_tokens=True).replace("\n", "").upper()
                if txt.startswith(target.epitope):
                    txt = txt[len(target.epitope):]
                txt = "".join(c for c in txt if c.isalpha())
                if not txt:
                    stuck_attempts += 1
                    continue
                header = f"protgpt2|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
                records.append((header, txt))
                if is_valid_vhh(txt):
                    valid_count += 1
                produced += 1
                if produced >= args.n:
                    break
        dt = time.time() - t0
        # incremental write after each target — the reason we're re-running
        write_fasta(out_fasta, records)
        write_status(args.out_dir, "protgpt2", {
            "n_targets": len(targets), "n_per_target": args.n,
            "targets_done": t_i + 1,
            "total_records": len(records), "valid_vhh": valid_count,
            "seconds": dt, "in_progress": True,
        })
        print(f"[protgpt2] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced}/{args.n} seqs  ({dt:.0f}s total, valid={valid_count})", flush=True)

    write_status(args.out_dir, "protgpt2", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "valid_vhh": valid_count,
        "seconds": time.time() - t0, "in_progress": False,
    })

if __name__ == "__main__":
    main()
