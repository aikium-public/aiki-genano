"""Sub-command dispatcher for ``python -m aiki_genano.cli``.

Manual dispatch (not argparse sub-parsers) so each sub-command can be
imported lazily — avoids paying a torch/transformers import cost just to
print --help on a different sub-command.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict

_USAGE = (
    "usage: aiki-genano <command> [args...]\n"
    "\n"
    "Aiki-GeNano CLI — sequence generation, property prediction, training, smoke tests.\n"
    "\n"
    "commands:\n"
    "  generate    Sample N candidate VHH sequences for an epitope; score with the\n"
    "              six GDPO reward functions. Writes predictions.csv.\n"
    "  predict     Compute the full property profile (TEMPRO, NetSolP, Sapiens, motifs)\n"
    "              for a list of sequences.\n"
    "  train       Run SFT / DPO / GDPO training, driven by a Hydra config.\n"
    "  smoke       End-to-end self-test of the image; --offline uses stubs,\n"
    "              --real pulls from Zenodo.\n"
    "\n"
    "Run 'aiki-genano <command> --help' for command-specific arguments.\n"
)


def _subcommands() -> Dict[str, Callable[[list[str]], int]]:
    def _generate(argv: list[str]) -> int:
        from aiki_genano.cli import generate
        return generate.main(argv)

    def _predict(argv: list[str]) -> int:
        from aiki_genano.cli import predict
        return predict.main(argv)

    def _train(argv: list[str]) -> int:
        from aiki_genano.cli import train
        return train.main(argv)

    def _smoke(argv: list[str]) -> int:
        from aiki_genano.cli import smoke
        return smoke.main(argv)

    return {
        "generate": _generate,
        "predict": _predict,
        "train": _train,
        "smoke": _smoke,
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        return 0

    cmds = _subcommands()
    cmd = argv[0]
    if cmd not in cmds:
        sys.stderr.write(f"aiki-genano: unknown command '{cmd}'\n\n")
        sys.stderr.write(_USAGE)
        return 2

    return cmds[cmd](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
