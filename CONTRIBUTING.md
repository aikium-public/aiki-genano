# Contributing to Aiki-GeNano

Thanks for your interest. Aiki-GeNano accompanies a published paper; the code and weights as released are intended to reproduce the paper. We welcome contributions in the following shape.

## What we welcome

- **Bug reports.** Open an issue with a minimal reproducer (epitope sequence, sampling parameters, exact image tag, observed vs expected behaviour). Smoke-test inside the published Docker image (`bash scripts/docker_smoke_test.sh --offline`) before reporting — it isolates environment problems.
- **Bug fixes.** Pull requests against `main`. Keep diffs focused (one issue per PR). Add or update one of the smoke tests if the fix changes a code path the smoke test exercises.
- **Documentation improvements.** README clarifications, schema corrections, additional FAQ entries — all welcome.
- **New evaluation predictors.** If you have a published nanobody / antibody developability predictor with a permissive licence, we are happy to wrap it under `aiki_genano/evaluation/` so it can be enabled via `predict --with-properties`. Open an issue first to discuss interface.

## What we cannot accept

- **Changes to the training data, the trained checkpoints, or the reward weights** — these are fixed by the paper. New experiments belong in a forked repo.
- **Contributions whose licence is incompatible with MIT** for code, or with CC-BY-NC-4.0 for the corresponding Zenodo data.

## Development setup

```bash
git clone https://github.com/aikium-public/aiki-genano.git
cd aiki-genano
pip install -e .[evaluation]
python -m aiki_genano.cli smoke --offline       # ~1 minute, no network, no GPU
```

For end-to-end testing with real checkpoints:

```bash
bash scripts/download_checkpoints.sh             # ~3.4 GB from Zenodo, sha256-verified
bash scripts/docker_smoke_test.sh --real --gpu   # ~10 minutes including model load
```

## Code style

- Python: run `python -m py_compile` on changed files; reward functions and CLI entrypoints should keep their existing argparse-style interfaces.
- Match the surrounding file's formatting conventions (no project-wide auto-formatter is enforced).
- Comments explain WHY, not WHAT — well-named identifiers handle the latter.

## Reporting security issues

For security-relevant issues (e.g. a vulnerability in a dependency, an unintended secret in the repo, or an attack vector against the deployed Modal endpoint), email `partnerships@aikium.com` rather than opening a public issue.

## Citation in derived work

If your contribution becomes part of a publication, please cite the original paper alongside this repo (citation snippet in `README.md`).
