# Benchmark comparison — Aiki-GeNano vs competing nanobody generators

Generated 2026-04-25 05:36 UTC. Seed 42, temperature 0.7, 100 seqs per epitope × 10 GPCR epitopes = 1000 per competitor.
**Cells in any row with `N valid < 10` are suppressed (shown as `—`)** because means on tiny gated subsamples are statistically meaningless and would mislead a reader scanning the table. ProtGPT2 in particular has only 2/900 loose-VHH-gate-passing sequences; its gated NetSolP values would read as decisive wins if reported and would be wrong.
Aiki-GeNano numbers from cached `data/aiki_genano_profiled/per_seed/{MODEL}_seed42_temp0.7_profiled.csv`; 
predictor self-consistency was verified before this run (re-running TEMPRO/NetSolP/Sapiens on 100 cached SFT seqs reproduced all 24 metrics within ±1 °C / ±0.05 NetSolP / exact match elsewhere).
**Read `CONFOUNDERS.md` before quoting any number from this table** — three confounders (locked vs. free scaffold, sample-size disparity, and 'are these even nanobodies?') materially affect interpretation.

Competitor sequences profiled via `benchmarks/run/profile_tool.py` using the same
`aiki_genano.evaluation.profile.compute_sequence_profile` pipeline + identical NetSolP ESM1b
setup. **Filtering**: Aiki-GeNano uses the paper-strict `is_valid_126` gate (126 AA +
GGGGS linker + ≥2 Cys + canonical alphabet). Competitors do not emit the GGGGS
engineering linker, so we report the loose gate `is_valid_vhh_loose` (110–130 AA + ≥2 Cys
+ canonical alphabet). Lengths, gate pass rates, and status are shown.

PepMLM generates 10-AA peptides, not VHHs; reported as peptide-only with VHH-specific
metrics (TEMPRO, Sapiens) intentionally blanked.

| Tool | Gate | N total | N valid | Tm (°C) | Instab. | Humanness | NetSolP sol | NetSolP use | Deam. | Isom. | GRAVY | Unique |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | is_valid_126 | 6400 | 6398 | 71.61 | 36.05 | 0.648 | 0.611 | 0.397 | 6.86 | 4.72 | -0.356 | 1.000 |
| DPO | is_valid_126 | 5998 | 5998 | 74.81 | 34.33 | 0.655 | 0.627 | 0.405 | 6.69 | 4.58 | -0.346 | 1.000 |
| GDPO(DPO) | is_valid_126 | 6386 | 6386 | 78.25 | 32.77 | 0.657 | 0.635 | 0.413 | 5.39 | 2.06 | -0.291 | 0.946 |
| GDPO(SFT) | is_valid_126 | 6399 | 6399 | 75.95 | 35.10 | 0.653 | 0.618 | 0.403 | 5.29 | 2.02 | -0.312 | 0.974 |
| nanoBERT | is_valid_vhh_loose | 1000 | 1000 | 65.94 | 31.63 | 0.706 | 0.577 | 0.399 | 6.36 | 4.19 | -0.395 | 1.000 |
| IgLM | is_valid_vhh_loose | 1000 | 997 | 68.27 | 34.08 | 0.740 | 0.587 | 0.425 | 6.82 | 7.18 | -0.348 | 1.000 |
| NanoAbLLaMA | is_valid_vhh_loose | 1000 | 933 | 73.21 | 32.94 | 0.750 | 0.598 | 0.434 | 6.37 | 4.56 | -0.315 | 1.000 |
| ProteinDPO | is_valid_vhh_loose | 1000 | 1000 | 56.31 | 28.92 | 0.591 | 0.630 | 0.403 | 6.77 | 6.17 | -0.298 | 1.000 |
| IgGM | is_valid_vhh_loose | 1000 | 1000 | 71.08 | 33.26 | 0.705 | 0.576 | 0.393 | 5.01 | 5.26 | -0.368 | 0.999 |
| ProtGPT2 | is_valid_vhh_loose | 900 | 2 | — | — | — | — | — | — | — | — | 1.000 |
| PepMLM | all (peptide) | 1000 | 1000 | — | 42.49 | — | 0.580 | 0.238 | 0.29 | 0.14 | 0.224 | 1.000 |

## Severity-weighted liability definitions

- Deamidation: `NG×3 + NS×3 + NT×2 + NN×2 + NH×2 + NA×1 + NQ×1`
- Isomerization: `DG×3 + DS×2 + DT×2 + DD×1 + DH×1`

Same weights used during GDPO reward training (see `aiki_genano/rewards/rewards.py`).

## Notes for readers

- NetSolP tolerance: fresh predictions differ from cached Aiki-GeNano values by up to
  ~0.03 per sequence (mean Δ < 0.01) due to `onnxruntime` version differences in
  quantized inference. Means are apples-to-apples; per-sequence ranks may shift slightly.
- CDR-level metrics in the profiled CSVs assume NBv1 scaffold positions (hard-coded IMGT
  offsets). Non-NBv1 competitors (everyone) get mechanical numbers that don't correspond
  to real CDRs — not shown here. Use global metrics only for cross-tool comparison.
- Any row showing `—` is genuinely missing (column absent or all-NaN), not zero.