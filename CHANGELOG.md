# Changelog

All notable changes to Aiki-GeNano are documented here. Versioning follows [SemVer](https://semver.org/).

## [1.0.0] — 2026-04-25

Initial public release accompanying the mAbs paper *"Preference-optimized generation of developable nanobodies across 65 epitope targets"* (Meda et al., 2026).

- Three-stage staged language-model alignment pipeline (SFT → DPO → GDPO) on ProtGPT2 for epitope-conditioned nanobody design.
- Six sequence-based GDPO reward functions: FR2 hydrophobicity, hydrophobic-patch coverage, chemical-liability motifs (deamidation/isomerisation/N-glycosylation/oxidation), Wilkinson–Harrison expression probability, VHH FR2 hallmark conservation, and scaffold integrity.
- Four trained checkpoints (SFT merged + DPO/GDPO_DPO/GDPO_SFT LoRA adapters) released on Zenodo.
- 10-target representative subset of the training data (10,000 sequences) and the matched DPO preference pairs (21,998 pairs) released on Zenodo, sufficient to verify the modelling claims.
- Single Docker image at `ghcr.io/aikium-public/aiki-genano:1.0.0` packages the SFT/DPO/GDPO training stack, the six reward functions, and the property-prediction pipeline (TEMPRO, NetSolP, Sapiens).
- Headline numbers vs the supervised baseline across the 65-target screen: predicted mean melting temperature +6.6 °C, deamidation severity reduced, N-glycosylation and CDR methionine-oxidation motif occurrence reduced; predicted humanness and solubility preserved. Generated sequences differ from the nearest training sequence by 8.1–9.0 amino acids out of 126.

[1.0.0]: https://github.com/aikium-public/aiki-genano/releases/tag/v1.0.0
