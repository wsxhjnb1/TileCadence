"""Solve shared-memory handoff offsets for Qwen3 attention."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Region:
    name: str
    size_bytes: int
    alignment_bytes: int
    start: int
    end: int


@dataclass(frozen=True)
class Handoff:
    donor_name: str
    donor_offset_bytes: int
    donor_bytes: int
    donor_release: int
    regions: tuple[Region, ...]
    timeline: tuple[str, ...]


HANDOFFS = {
    "prefill": Handoff(
        "terminal_K", 64 * 1024, 16 * 1024, 1,
        (Region("head0_lse", 512, 128, 1, 3),
         Region("head1_lse", 512, 128, 1, 3)),
        ("last_K_read", "K_readers_released_and_epilogue_scatter", "CTA_publish_and_export"),
    ),
    "decode": Handoff(
        "terminal_V", 0, 32 * 1024, 1,
        (Region("output_tile", 2048, 16, 1, 3),
         Region("row_lse", 16, 16, 1, 3)),
        ("last_V_read", "V_readers_released_and_epilogue_scatter", "CTA_publish_and_export"),
    ),
}


def replay(handoff: Handoff, offsets: dict[str, int]) -> None:
    """Independently check half-open times and byte ranges."""
    donor_lo = handoff.donor_offset_bytes
    donor_hi = donor_lo + handoff.donor_bytes
    regions = handoff.regions
    for i, region in enumerate(regions):
        lo = offsets[region.name]
        hi = lo + region.size_bytes
        if not (handoff.donor_release <= region.start < region.end <= len(handoff.timeline)
                and donor_lo <= lo < hi <= donor_hi and lo % region.alignment_bytes == 0):
            raise ValueError(f"Invalid time/address placement for {region.name}")
        for other in regions[:i]:
            if max(region.start, other.start) < min(region.end, other.end):
                other_lo = offsets[other.name]
                if max(lo, other_lo) < min(hi, other_lo + other.size_bytes):
                    raise ValueError(f"Live byte ranges overlap for {region.name} and {other.name}")


def solve(handoff: Handoff) -> dict:
    model = cp_model.CpModel()
    donor_lo = handoff.donor_offset_bytes
    donor_hi = donor_lo + handoff.donor_bytes
    high_water = model.new_int_var(donor_lo, donor_hi, "recipient_high_water")
    time_spans, byte_spans, offsets = [], [], {}
    for region in handoff.regions:
        step = region.alignment_bytes
        first = (donor_lo + step - 1) // step
        last = (donor_hi - region.size_bytes) // step
        slot = model.new_int_var(first, last, f"{region.name}_slot")
        offset = model.new_int_var(first * step, last * step, f"{region.name}_bytes")
        model.add(offset == step * slot)
        model.add(high_water >= offset + region.size_bytes)
        time_spans.append(model.new_fixed_size_interval_var(
            region.start, region.end - region.start, f"{region.name}_time"))
        byte_spans.append(model.new_fixed_size_interval_var(
            offset, region.size_bytes, f"{region.name}_address"))
        offsets[region.name] = offset
    model.add_no_overlap_2d(time_spans, byte_spans)
    # Compact the recipient span and break ties deterministically.
    model.minimize((donor_hi + 1) * high_water + offsets[handoff.regions[0].name])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    if solver.solve(model) != cp_model.OPTIMAL:
        raise ValueError("No optimal physical handoff placement")
    placement = {name: solver.value(value) for name, value in offsets.items()}
    replay(handoff, placement)
    return {
        "schema": "qwen3-residency-placement-v79",
        "status": "OPTIMAL",
        "donor": {"name": handoff.donor_name, "offset_bytes": donor_lo,
                  "size_bytes": handoff.donor_bytes, "release": handoff.donor_release},
        "regions": {r.name: {**asdict(r), "offset_bytes": placement[r.name]}
                    for r in handoff.regions},
        "timeline": handoff.timeline,
        "recipient_high_water_bytes": solver.value(high_water),
        "objective": "Minimum recipient address high-water, then first recipient offset",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=HANDOFFS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for phase in (args.phase,) if args.phase else HANDOFFS:
        with (args.out / f"{phase}.json").open("x", encoding="utf-8") as stream:
            json.dump(solve(HANDOFFS[phase]), stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
