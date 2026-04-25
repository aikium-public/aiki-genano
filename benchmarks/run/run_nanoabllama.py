"""NanoAbLLaMA runner: prompted VHH generation via Seq= tag.

Output: benchmark_out/nanoabllama/generated_T0.7_seed{seed}.fasta

Prompt: `Seq=<QVQLV` — the best one from our smoke tests. Epitope noted in
header only (NanoAbLLaMA has no epitope conditioning).

Requires HF_TOKEN in env for the gated Lab608/NanoAbLLaMA.
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
        print(f"[nanoabllama] already have {total_expected}; skipping"); return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[nanoabllama] ERROR: HF_TOKEN not set", file=sys.stderr); sys.exit(1)

    random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained("Lab608/NanoAbLLaMA", token=token)
    mdl = AutoModelForCausalLM.from_pretrained(
        "Lab608/NanoAbLLaMA", token=token,
        quantization_config=bnb, device_map={"": 0},
    ).eval()
    print(f"[nanoabllama] model ready", flush=True)

    # Best prompt from our smoke tests
    prompt = "Seq=<QVQLV"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)
    attn = torch.ones_like(prompt_ids)

    records = []
    valid_count = 0
    t0 = time.time()
    batch_size = 4   # 4-bit LLaMA-7B, be conservative on 16 GB shared GPU
    for t_i, target in enumerate(targets):
        produced = 0
        while produced < args.n:
            k = min(batch_size, args.n - produced)
            try:
                with torch.no_grad():
                    out = mdl.generate(
                        prompt_ids.repeat(k, 1),
                        attention_mask=attn.repeat(k, 1),
                        max_new_tokens=180,
                        do_sample=True, temperature=args.T, top_p=0.9,
                        num_return_sequences=1,
                        pad_token_id=tok.eos_token_id,
                    )
            except Exception as e:
                print(f"[nanoabllama] gen err {target.uniprot_id}: {e}", file=sys.stderr)
                break
            for o in out:
                txt = tok.decode(o, skip_special_tokens=True)
                # Text format: "Seq=<QVQLV... sequence here ...>"
                if "Seq=<" in txt:
                    body = txt.split("Seq=<", 1)[1]
                else:
                    body = txt[len(prompt):]
                # Terminate at closing bracket or newline
                body = body.split(">")[0].split("\n")[0]
                seq = "".join(c for c in body if c.isalpha()).upper()
                if not seq:
                    continue
                header = f"nanoabllama|{target.uniprot_id}|{target.name.replace(' ','_')}|{produced}"
                records.append((header, seq))
                if is_valid_vhh(seq):
                    valid_count += 1
                produced += 1
                if produced >= args.n:
                    break
        dt = time.time() - t0
        # incremental write after every target so a stall doesn't wipe everything
        write_fasta(out_fasta, records)
        write_status(args.out_dir, "nanoabllama", {
            "n_targets": len(targets), "n_per_target": args.n,
            "targets_done": t_i + 1,
            "total_records": len(records), "valid_vhh": valid_count,
            "seconds": dt, "in_progress": True,
        })
        print(f"[nanoabllama] {t_i+1}/{len(targets)} {target.uniprot_id}: {produced}  ({dt:.0f}s, valid={valid_count})", flush=True)

    write_status(args.out_dir, "nanoabllama", {
        "n_targets": len(targets), "n_per_target": args.n,
        "total_records": len(records), "valid_vhh": valid_count,
        "seconds": time.time() - t0, "in_progress": False,
    })

if __name__ == "__main__":
    main()
