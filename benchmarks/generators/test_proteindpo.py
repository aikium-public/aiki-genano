"""ProteinDPO smoke test.

ProteinDPO = ESM-IF1 + DPO-finetuned weights (backbone-conditioned inverse
folding). The upstream repo lives at https://github.com/evo-design/protein-dpo.
It does NOT ship as a pip package — clone the repo, apply the patch in
../patches/proteindpo_sample.patch (a single-line fix: current fair-esm
rejects the `device=` kwarg that upstream still passes to `model.sample()`),
then invoke sample.py directly.

Env: .venv-bench with `fair-esm`, `biotite<0.40`, `torch-scatter` matched to
torch version. See ../INSTALLED.md §2.5.

Example invocation (run against a canonical VHH backbone PDB):

    CUDA_VISIBLE_DEVICES="" /path/to/.venv-bench/bin/python \
        protein-dpo/sample.py \
        --pdbfile path/to/vhh_backbone.pdb \
        --chain H \
        --num-samples 3 \
        --temperature 0.7 \
        --outpath example_outputs/proteindpo/smoke.fasta

The CPU flag is a workaround for a device-placement bug in the current esm
package (coords end up on CPU while the model is on GPU); upstream fix is to
cast `coords` to cuda before `.sample()`.
"""
