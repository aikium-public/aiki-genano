"""``aiki-genano train`` — Hydra-driven training launcher.

Thin wrapper over ``aiki_genano.training.{sft,dpo,gdpo}`` that resolves the
right config_name based on ``--stage`` and forwards the rest of argv to the
Hydra @main entrypoint. Supports the smoke-config shortcut (``--smoke``).

Examples:
    aiki-genano train --stage sft --config sft_10k
    aiki-genano train --stage dpo --config dpo_developability
    aiki-genano train --stage gdpo --config gdpo/dpo_final_gated
    aiki-genano train --stage sft --smoke         # configs/smoke/sft.yaml, 10 steps
    aiki-genano train --stage gdpo --config gdpo/dpo_final_gated training_args.max_steps=100

Hydra overrides (key=value) after --config flow through unchanged; same for
--config-path.
"""
from __future__ import annotations

import argparse
import sys
from typing import Tuple


_STAGE_ENTRYPOINTS = {
    "sft":  "aiki_genano.training.sft",
    "dpo":  "aiki_genano.training.dpo",
    "gdpo": "aiki_genano.training.gdpo",
}

_STAGE_DEFAULT_CONFIGS = {
    "sft":  "sft_10k",
    "dpo":  "dpo_developability",
    "gdpo": "gdpo/dpo_final_gated",
}

_STAGE_SMOKE_CONFIGS = {
    "sft":  "smoke/sft",
    "dpo":  "smoke/dpo",
    "gdpo": "smoke/gdpo",
}


def _split(argv: list[str]) -> Tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        prog="aiki-genano train",
        description=(
            "Run SFT / DPO / GDPO training using Hydra configs from configs/. "
            "Pass Hydra overrides (key=value) as positional args after the flags."
        ),
        add_help=True,
    )
    p.add_argument("--stage", required=True, choices=sorted(_STAGE_ENTRYPOINTS.keys()),
                   help="Training stage to run.")
    p.add_argument("--config", default=None,
                   help="Config name (relative to configs/, no .yaml extension). "
                        "Default: stage-specific (sft_10k / dpo_developability / "
                        "gdpo/dpo_final_gated).")
    p.add_argument("--config-path", default="../../configs",
                   help="Hydra config_path override (default: ../../configs).")
    p.add_argument("--smoke", action="store_true",
                   help=("Use the smoke config for the stage (10 steps, batch 2, no eval). "
                         "Mutually exclusive with --config."))
    return p.parse_known_args(argv)


def main(argv: list[str]) -> int:
    args, hydra_overrides = _split(argv)

    if args.smoke and args.config:
        raise SystemExit("--smoke and --config are mutually exclusive.")

    if args.smoke:
        config_name = _STAGE_SMOKE_CONFIGS[args.stage]
    elif args.config:
        config_name = args.config
    else:
        config_name = _STAGE_DEFAULT_CONFIGS[args.stage]

    module = _STAGE_ENTRYPOINTS[args.stage]
    print(f"[train] stage={args.stage} module={module} "
          f"config={config_name} overrides={hydra_overrides}")

    # Hand off to the stage's @hydra.main entrypoint by rewriting sys.argv,
    # then importing and calling main(). The Hydra decorator reads sys.argv
    # so we need it to look like a direct `python -m <module>` invocation.
    sys.argv = [
        f"python -m {module}",
        f"--config-path={args.config_path}",
        f"--config-name={config_name}",
        *hydra_overrides,
    ]

    import importlib
    mod = importlib.import_module(module)
    if not hasattr(mod, "main"):
        raise SystemExit(f"FATAL: {module} has no main() — wiring out of date.")

    # The Hydra-decorated main() takes no args; Hydra reads sys.argv.
    mod.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
