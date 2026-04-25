"""Command-line entrypoints for the Aiki-GeNano Docker image.

Sub-commands are routed by ``aiki_genano/cli/__main__.py`` to per-command
implementations under this package. The Dockerfile sets ``ENTRYPOINT`` to
``python -m aiki_genano.cli`` so users invoke them as e.g.::

    docker run ... aiki-genano:latest generate --epitope … --n_candidates 50
    docker run ... aiki-genano:latest predict  --sequences seq.fasta --with-properties
    docker run ... aiki-genano:latest train    --stage sft --config configs/sft_subset.yaml
    docker run ... aiki-genano:latest smoke    --offline
"""
