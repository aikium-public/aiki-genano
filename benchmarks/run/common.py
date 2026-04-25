"""Shared helpers for benchmark runners."""
import csv, os, json, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# VHH template (camelid) — used by tools that need a canonical backbone
# with CDRs marked as X for infilling. Matches IgGM CDR_All shape.
VHH_TEMPLATE_ALL_X = (
    "QVQLVESGGDLVQSGGSLKLACAVS"        # 1-25 FR1
    ("X" * 7)                            # 26-32 CDR1 (mask for infilling)
    "SIGWFRQAPGKEREAVSYS"                # 33-51 FR2
    ("X" * 6)                            # 52-57 CDR2 (mask for infilling)
    "TYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAA"  # 58-98 FR3
    ("X" * 18)                           # 99-116 CDR3 (mask for infilling)
    "WGQGTQVTVSS"                        # 117-127 FR4
)
# Same backbone fully materialised for nanoBERT / MLM-style fill
VHH_TEMPLATE_FILLED = (
    "QVQLVESGGDLVQSGGSLKLACAVS"
    "GFTFSSY"                            # generic CDR1 fill (IgLM-consensus)
    "SIGWFRQAPGKEREAVSYS"
    "INSGGG"                             # generic CDR2 fill
    "TYYVASVKGRFTISRDNAKNTAYLQMNNLKPEDTGIYYCAA"
    ("A" * 18)                           # CDR3 stub fill — re-masked downstream
    "WGQGTQVTVSS"
)
CDR1_POSITIONS = list(range(25, 32))    # 7 positions
CDR2_POSITIONS = list(range(51, 57))    # 6 positions
CDR3_POSITIONS = list(range(98, 116))   # 18 positions
ALL_CDR_POSITIONS = CDR1_POSITIONS + CDR2_POSITIONS + CDR3_POSITIONS

@dataclass
class Target:
    uniprot_id: str
    name: str
    epitope: str

def load_targets(csv_path: str) -> list[Target]:
    targets = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            targets.append(Target(
                uniprot_id=row["UniProt_ID"].strip(),
                name=row["Target Name"].strip(),
                epitope=row["epitope"].strip().upper(),
            ))
    return targets

def write_fasta(path: str, records: list[tuple[str, str]]):
    """records: list of (header, sequence). Header will be prefixed with '>'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for h, s in records:
            f.write(f">{h}\n{s}\n")

def write_status(out_dir: str, tool: str, status: dict):
    status["tool"] = tool
    status["timestamp"] = time.time()
    status["time_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = os.path.join(out_dir, "status.json")
    with open(path, "w") as f:
        json.dump(status, f, indent=2)

def already_done(out_fasta: str, n_expected: int) -> bool:
    """Returns True if out_fasta exists with at least n_expected records."""
    if not os.path.exists(out_fasta):
        return False
    with open(out_fasta) as f:
        count = sum(1 for line in f if line.startswith(">"))
    return count >= n_expected

AA_CANON = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_vhh(seq: str, min_len=110, max_len=135) -> bool:
    if not (min_len <= len(seq) <= max_len):
        return False
    if set(seq) - AA_CANON:
        return False
    if seq.count("C") < 2:
        return False
    return True

def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:8]
