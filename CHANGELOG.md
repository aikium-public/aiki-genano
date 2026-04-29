# Changelog

All notable changes to Aiki-GeNano are documented here. Versioning follows [SemVer](https://semver.org/).

## [1.1.0] — 2026-04-28

Manuscript-revision sync. Source code, training pipeline, Docker image, and Zenodo data files are unchanged from 1.0.0; this release tracks paper, README, and Modal-landing updates.

- README: removed an internal-only Google Cloud Storage section that should not have been on the public surface; corrected the "Pre-trained model checkpoints" section to state that all four checkpoints are NDA-gated via `partnerships@aikium.com` (the previous wording incorrectly suggested a public GCS bucket); rewrote "Reproducing paper numbers" against what the Zenodo deposit actually ships (`figure_data/`, `full_property_tables/`, `PROVENANCE.md`); broadened the FAQ "any target" answer to mention soluble domains alongside peptides.
- Modal landing (`web/index.html`): re-pasted the paper abstract verbatim against the revised submission (now names "Aiki-GeNano" explicitly, lists selectivity in the DPO composite, capitalises GDPO as "Group Reward-Decoupled Policy Optimization", and adds the head-to-head sentence "On a shared 10-target GPCR benchmark, the pipeline achieved the highest predicted melting temperature and lowest isomerization severity among five contemporary VHH generators"); added a hero stat tile reflecting the head-to-head result; tightened the Limits panel "Works well" card to mention the head-to-head comparators by name; added Therapeutic Nanobody Profiler (Gordon 2026, Oxford OPIG) to the credits panel.
- Acknowledgements (this README + landing): added Therapeutic Nanobody Profiler reference and the five named head-to-head comparators.

## [1.0.0] — 2026-04-25

Initial public release accompanying the mAbs (submitted) paper *"Aiki-GeNano: Multi-Stage Preference Optimization for Generative Design of Developable Nanobodies"* (Meda et al., 2026).

- Three-stage staged language-model alignment pipeline (SFT → DPO → GDPO) on ProtGPT2 for epitope-conditioned nanobody design.
- Six sequence-based GDPO reward functions: FR2 hydrophobicity, hydrophobic-patch coverage, chemical-liability motifs (deamidation/isomerisation/N-glycosylation/oxidation), Wilkinson–Harrison expression probability, VHH FR2 hallmark conservation, and scaffold integrity.
- Single Docker image at `ghcr.io/aikium-public/aiki-genano:1.0.0` packages the SFT/DPO/GDPO training stack, the six reward functions, and the property-prediction pipeline (TEMPRO, NetSolP, Sapiens).
- Modal-hosted live demo at <https://aikium--aiki-genano-fastapi-app.modal.run> running the GDPO_DPO checkpoint (`/api/generate`, `/api/score`, `/api/health`).
- Numerical figure data + per-sequence property profiles (sequences stripped) deposited at Zenodo under CC-BY-NC-4.0: <https://doi.org/10.5281/zenodo.19757842>.
- Trained model checkpoints, the underlying nanobody-screening dataset, and the literal generated nanobody sequences remain proprietary; available from `partnerships@aikium.com` under NDA.
- Headline numbers vs the supervised baseline across the 65-target screen: predicted mean melting temperature +6.6 °C, deamidation severity reduced, N-glycosylation and CDR methionine-oxidation motif occurrence reduced; predicted humanness and solubility preserved. Generated sequences differ from the nearest training sequence by 8.1–9.0 amino acids out of 126.

[1.1.0]: https://github.com/aikium-public/aiki-genano/releases/tag/v1.1.0
[1.0.0]: https://github.com/aikium-public/aiki-genano/releases/tag/v1.0.0
