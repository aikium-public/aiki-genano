#!/usr/bin/env bash
#
# Aiki-GeNano model checkpoints (SFT, DPO, GDPO_DPO, GDPO_SFT) are NOT
# redistributed in the public Zenodo deposit at
#   https://doi.org/10.5281/zenodo.19757843
# Only the numerical figure data and per-sequence computed property profiles
# (with amino-acid sequences stripped) are deposited there.
#
# To obtain the trained model checkpoints, contact partnerships@aikium.com.
# Aikium will share them under a non-disclosure agreement for academic
# evaluation, custom design campaigns, or licensed deployments.
#
# Once you have the four checkpoint directories on disk, place them as:
#   ./checkpoints/SFT/NanoBody-design-sft-response-only-100k-len126-r64/
#   ./checkpoints/DPO/checkpoint-6000/
#   ./checkpoints/GDPO_DPO/checkpoint-2000/
#   ./checkpoints/GDPO_SFT/checkpoint-2000/
# and the Docker image's `aiki-genano generate --model {SFT,DPO,GDPO_DPO,GDPO_SFT}`
# CLI will pick them up automatically.
#
# To play with the model immediately without setting up checkpoints locally,
# use the public Modal demo:
#   https://aikium--aiki-genano-fastapi-app.modal.run

set -e

cat <<'EOF' >&2
Trained checkpoints are not in the public Zenodo deposit.

  - Public Zenodo deposit (figure data only):
      https://doi.org/10.5281/zenodo.19757843

  - Checkpoints (NDA-gated, request from Aikium):
      partnerships@aikium.com

  - Live demo (no checkpoints needed):
      https://aikium--aiki-genano-fastapi-app.modal.run

This script intentionally exits without downloading anything.
EOF

exit 0
