# Confounders that affect interpretation of `BENCHMARK_COMPARISON.md`

Anyone using the head-to-head numbers in a paper, slide, or claim **must
address these three confounders** in the text. Each is supported by the data;
each materially changes how the table should be read.

---

## 1. Aiki-GeNano is locked into one scaffold; competitors are not

Aiki-GeNano's training pipeline anchors every output to the NBv1 scaffold.
Empirically, **103 / 126 positions are completely invariant** across all
6,386 valid GDPO(DPO) sequences. 100 % of them start with
`QVQLVESGGGSVQAGGSLRLSCTAS` and end with `…WGQGTQVTVS GGGGS`. Generative
variation is concentrated in the 23 CDR + adjacent positions.

By contrast NanoAbLLaMA varies at 100 / 126 positions across 933 sequences,
and IgLM / IgGM are similarly diverse on the framework axis.

| Tool | Invariant positions (1 char across the gated subset) | High-diversity positions (>5 chars) |
|---|---|---|
| GDPO(DPO) | 103 / 126 (82 %) | 9 / 126 |
| NanoAbLLaMA | 26 / 126 (21 %) | 98 / 126 |

**What this does to the headline numbers**

- Aiki-GeNano's T<sub>m</sub>, instability, GRAVY, NetSolP solubility, and
  isomerization severity are largely *properties of the locked scaffold*
  with marginal CDR contribution. The 5 °C T<sub>m</sub> Aiki-vs-NanoAbLLaMA
  gap is partly "the hand-picked Aiki scaffold has a higher prior
  T<sub>m</sub> than the average camelid VHH NanoAbLLaMA samples."
- Competitor metrics, conversely, average over **many distinct frameworks**.
  A meaningful comparison would be Aiki-GeNano on its scaffold vs each
  competitor restricted to its single best-T<sub>m</sub> framework — but
  competitors don't expose that lever.
- The very low σ_isomerization for GDPO (0.24) vs nanoBERT (2.0) and IgLM
  (2.6) is partly "GDPO has only 23 positions where motifs can appear" not
  "GDPO chose biochemistry-better residues at every position." Some of the
  σ shrinkage is mechanical.

**Recommended paper framing** *(both directions, defensible)*: "Aiki-GeNano's
optimisation operates within a chosen high-T<sub>m</sub>, low-isomerization
scaffold and improves it further. Competitors sample from a broader, more
humanness-rich framework distribution at the cost of these biophysical
axes."

---

## 2. Sample-size disparity (Aiki ~6× larger pool)

Aiki-GeNano profiled CSVs each contain ~6,386 valid sequences (one seed,
unconditional generation, 6 500 candidates × ~98 % strict-gate pass rate).
Competitors are 1 000 conditional generations each (10 epitopes × 100 seqs)
with ~93-100 % loose-gate pass rate, so each competitor mean is over
933-1 000 sequences. **Aiki has ~6× more sequences in every mean.**

Practical impact:

- **Standard error of the mean** for T<sub>m</sub>: GDPO(DPO) SE = 0.04 °C
  vs NanoAbLLaMA SE = 0.19 °C. Both far below the 5 °C effect size, so the
  disparity does NOT alter pairwise significance — every reported gap is
  comfortably ≥10× SE on both sides.
- The disparity DOES affect the σ values reported in `mean ± σ` columns.
  Aiki's σ (e.g. 3.6 °C for T<sub>m</sub>) is partly the scaffold's
  intrinsic prediction range; competitor σ (5.9 °C) is partly framework
  diversity. Don't read σ as "predictor noise" — read it as "spread across
  whatever the tool's generation distribution produced." Confounder #1
  drives most of this gap.
- Aiki has 1 seed × 6 500 candidates; competitors have 10 epitopes × 100
  seqs but only seed = 42. Multi-seed variance is uncharted on the
  competitor side. The paper text should state both N's openly.

---

## 3. Are competitor outputs even nanobodies?

The loose `is_valid_vhh_loose` gate (110-130 AA + ≥2 Cys + canonical
alphabet) is permissive — it accepts non-VHH proteins of the right length
and Cys count. Stricter VHH-shape checks reveal heterogeneity:

| Tool | Loose gate | VHH-like N-term `^[QE]V[QK]L` | Canonical FR4 `WG[QR]GT` | Verdict |
|---|---|---|---|---|
| Aiki-GeNano (any) | 6386 | 6386/6386 (100 %) | 6385/6386 (100 %) | canonical VHH |
| nanoBERT | 1000 | **1000/1000 (100 %)** | **1000/1000 (100 %)** | canonical VHH |
| IgGM | 1000 | **1000/1000 (100 %)** | **1000/1000 (100 %)** | canonical VHH |
| NanoAbLLaMA | 933 | 933/933 (100 %) | 795/933 (85 %) | mostly canonical |
| IgLM | 997 | 13/997 (1.3 %) | 908/997 (91 %) | non-canonical N-term but FR4 mostly OK |
| **ProteinDPO** | 1000 | **20/1000 (2 %)** | **6/1000 (0.6 %)** | **NOT canonical VHHs** |
| ProtGPT2 | 2 | (n too small) | (n too small) | not a VHH-class generator |
| PepMLM | 0 (peptide-only) | n/a | n/a | 10 AA peptides, not nanobodies |

**Implications**

- ProteinDPO outputs satisfy the loose gate (length + 2 Cys + alphabet) but
  are **not VHHs in any structural sense**. They start with arbitrary
  residues (`MVT`, `GVT`, `NVRL`, …) and have no `WGQGTQVT` FR4 hallmark.
  This explains its anomalously low predicted T<sub>m</sub> (TEMPRO trained
  on VHHs sees out-of-distribution inputs, σ_T<sub>m</sub> = 9.4 vs 3.6-5.9
  for true VHHs) and why "ProteinDPO has best instability" should be
  qualified — it is a stability-optimised novel-sequence backbone, not a
  stability-optimised nanobody.
- IgLM's "non-canonical N-terminus" (1.3 % match `^[QE]V[QK]L`) is because
  IgLM strips the canonical N-terminal `Q` in its sampling — its sequences
  start with `VQL…` (V at pos 0, where Aiki has Q-V-Q-L…). Single missing
  residue; rest of the sequence is camelid VHH; treat IgLM as
  canonical-equivalent.
- ProtGPT2 (median 20 AA, 2 / 900 pass loose VHH gate) and PepMLM (10 AA
  peptides) are not VHH-class generators with our prompt strategy and
  should be excluded from VHH head-to-head claims.

**Recommended paper handling**

- Add a "VHH validity" column or footnote to the head-to-head table showing
  the canonical-FR4 % per tool.
- Treat ProteinDPO with a † footnote: *"ProteinDPO outputs satisfy a
  permissive length+Cys validity gate but do not match canonical VHH
  framework structure (only 0.6 % have a canonical `WG[QR]GT` FR4 motif).
  Its developability metrics are reported for completeness but should not
  be read as 'nanobody developability'."*
- Treat ProtGPT2 and PepMLM as control / scope-clarification entries, not
  as comparators.

---

## tl;dr table for the paper text

| # | Confounder | Where it bites | Mitigation in paper text |
|---|---|---|---|
| 1 | Aiki = 1 scaffold; competitors = many | T<sub>m</sub>, instability, isomerization, GRAVY, σ values | "Aiki improves within a hand-picked scaffold; competitors sample broader frameworks" |
| 2 | Aiki has 6× more sequences | σ ratios in mean ± σ; multi-seed coverage | State N for every row; SEs are tiny so significance unaffected |
| 3 | Some "competitors" aren't VHHs | ProteinDPO (0.6 % FR4), ProtGPT2 (2/900), PepMLM (peptide) | Footnote VHH-validity %; flag/exclude non-VHH-class outputs |
