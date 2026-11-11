from pathlib import Path

import pytest

from tilecadence.operator import OFFSETS, emit
from tilecadence.placement import HANDOFFS, replay


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_solved_offsets_reach_the_cuda_template(tmp_path: Path, phase: str):
    placement = emit(phase, tmp_path / phase)
    config = (tmp_path / phase / "config.cuh").read_text()
    offsets = {name: value["offset_bytes"] for name, value in placement["regions"].items()}
    for name, macro in OFFSETS[phase].items():
        assert f"#define {macro} {offsets[name]}\n" in config
    first, second = offsets
    offsets[second] = offsets[first]
    with pytest.raises(ValueError, match="overlap"):
        replay(HANDOFFS[phase], offsets)
