# Per-target means and winner — VHH-class competitors only

Aiki-GeNano variants are unconditional (no per-target structure); included only at the global level in `BENCHMARK_COMPARISON.md`.

## TEMPRO Tm (°C) — higher is better

| target_name                     |   IgGM |   IgLM |   NanoAbLLaMA |   ProteinDPO |   nanoBERT | WINNER      |
|:--------------------------------|-------:|-------:|--------------:|-------------:|-----------:|:------------|
| C-C_chemokine_receptor_type_2   |  69.2  |  68.74 |         72.35 |        54.79 |      65.37 | NanoAbLLaMA |
| C-C_chemokine_receptor_type_3   |  71.02 |  68.4  |         72.79 |        56.99 |      66.5  | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_1 |  72.68 |  68.24 |         72.2  |        55.35 |      65.69 | IgGM        |
| C-X-C_chemokine_receptor_type_3 |  72.47 |  68.61 |         74.23 |        55.7  |      65.49 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_4 |  70.85 |  67.03 |         73.09 |        57.54 |      65.91 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_5 |  70.94 |  69    |         73.46 |        55.86 |      66.19 | NanoAbLLaMA |
| Cannabinoid_receptor_1          |  69.2  |  67.76 |         72.99 |        56.08 |      66.31 | NanoAbLLaMA |
| Histamine_H4_receptor           |  72.35 |  68.29 |         73.77 |        56.21 |      65.44 | NanoAbLLaMA |
| Kappa-type_opioid_receptor      |  69.33 |  68.26 |         73.56 |        56.85 |      66.22 | NanoAbLLaMA |
| Mu-type_opioid_receptor         |  72.78 |  68.33 |         73.58 |        57.72 |      66.33 | NanoAbLLaMA |

## Instability index — lower is better

| target_name                     |   IgGM |   IgLM |   NanoAbLLaMA |   ProteinDPO |   nanoBERT | WINNER     |
|:--------------------------------|-------:|-------:|--------------:|-------------:|-----------:|:-----------|
| C-C_chemokine_receptor_type_2   |  35.3  |  34.36 |         33.59 |        27.69 |      31.04 | ProteinDPO |
| C-C_chemokine_receptor_type_3   |  34.22 |  34.53 |         31.94 |        28.84 |      31.76 | ProteinDPO |
| C-X-C_chemokine_receptor_type_1 |  34.32 |  35.01 |         32.62 |        31.36 |      31.84 | ProteinDPO |
| C-X-C_chemokine_receptor_type_3 |  32.46 |  33.41 |         32.51 |        30.59 |      31.61 | ProteinDPO |
| C-X-C_chemokine_receptor_type_4 |  36.01 |  34.86 |         32.19 |        29.75 |      31.44 | ProteinDPO |
| C-X-C_chemokine_receptor_type_5 |  32.17 |  33.93 |         34.25 |        28.06 |      32.24 | ProteinDPO |
| Cannabinoid_receptor_1          |  32.58 |  33.65 |         33.29 |        28.05 |      31.83 | ProteinDPO |
| Histamine_H4_receptor           |  32.72 |  33.83 |         33.45 |        28.53 |      32.05 | ProteinDPO |
| Kappa-type_opioid_receptor      |  31.35 |  33.34 |         32.93 |        28.21 |      31.7  | ProteinDPO |
| Mu-type_opioid_receptor         |  31.47 |  33.9  |         32.71 |        28.15 |      30.79 | ProteinDPO |

## Sapiens humanness — higher is better

| target_name                     |   IgGM |   IgLM |   NanoAbLLaMA |   ProteinDPO |   nanoBERT | WINNER      |
|:--------------------------------|-------:|-------:|--------------:|-------------:|-----------:|:------------|
| C-C_chemokine_receptor_type_2   |  0.702 |  0.739 |         0.752 |        0.591 |      0.705 | NanoAbLLaMA |
| C-C_chemokine_receptor_type_3   |  0.705 |  0.741 |         0.748 |        0.59  |      0.707 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_1 |  0.71  |  0.741 |         0.75  |        0.591 |      0.707 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_3 |  0.707 |  0.74  |         0.752 |        0.593 |      0.707 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_4 |  0.705 |  0.742 |         0.752 |        0.593 |      0.706 | NanoAbLLaMA |
| C-X-C_chemokine_receptor_type_5 |  0.708 |  0.734 |         0.749 |        0.592 |      0.706 | NanoAbLLaMA |
| Cannabinoid_receptor_1          |  0.707 |  0.744 |         0.751 |        0.59  |      0.706 | NanoAbLLaMA |
| Histamine_H4_receptor           |  0.701 |  0.739 |         0.748 |        0.592 |      0.707 | NanoAbLLaMA |
| Kappa-type_opioid_receptor      |  0.705 |  0.735 |         0.751 |        0.591 |      0.707 | NanoAbLLaMA |
| Mu-type_opioid_receptor         |  0.701 |  0.742 |         0.75  |        0.591 |      0.707 | NanoAbLLaMA |

## Tally

| Metric | Winner counts (out of 10 targets) |
|---|---|
| `tempro_tm` | {'NanoAbLLaMA': 9, 'IgGM': 1} |
| `instability_index` | {'ProteinDPO': 10} |
| `sapiens_humanness` | {'NanoAbLLaMA': 10} |
