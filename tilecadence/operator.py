"""Emit the prefill or decode schedule with solved shared-memory offsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from .placement import HANDOFFS, solve


OFFSETS = {
    "prefill": {"head0_lse": "PREFILL_LSE_HEAD0_OFFSET_BYTES",
                "head1_lse": "PREFILL_LSE_HEAD1_OFFSET_BYTES"},
    "decode": {"output_tile": "AB_DECODE_O_OFFSET_BYTES",
               "row_lse": "AB_DECODE_LSE_OFFSET_BYTES"},
}


def emit(phase: str, out: Path) -> dict:
    if phase not in OFFSETS:
        raise ValueError(f"Unknown operator: {phase}")
    placement = solve(HANDOFFS[phase])
    template = Path(__file__).parent / "cuda" / phase
    config = (template / "config.cuh").read_text(encoding="utf-8")
    for region, macro in OFFSETS[phase].items():
        offset = placement["regions"][region]["offset_bytes"]
        config, count = re.subn(rf"^#define {macro} \d+$", f"#define {macro} {offset}", config, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one template definition of {macro}")
    out.mkdir(parents=True, exist_ok=False)
    for path in template.iterdir():
        if path.name != "config.cuh":
            shutil.copyfile(path, out / path.name)
    (out / "config.cuh").write_text(config, encoding="utf-8", newline="\n")
    (out / "placement.json").write_text(json.dumps(placement, indent=2) + "\n", encoding="utf-8")
    return placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=OFFSETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    emit(args.phase, args.out)


if __name__ == "__main__":
    main()
