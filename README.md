# TileCadence

TileCadence generates FlashAttention-2 CUDA schedules that free shared-memory
regions for other intermediate values. A released region can hold a larger
tile or several smaller values that need to stay live at the same time.
The planner models byte ranges, alignment, consumers, and barriers before
the CUDA/CuTe templates receive the resulting offsets.

| Schedule | Shared-memory reuse |
|---|---|
| Two-tile attention | Release a 16 KiB Q tile, reload Q elsewhere for the next QK, and keep two 4 KiB probability checkpoints in the released range until their PV consumers. An alternative places one probability checkpoint alongside a V stripe. |
| Causal prefill | Use the terminal K region for two head-local LSE arrays and publish them at the existing O writeback barrier. |
| Fused decode | Use the terminal V region for the output rows and LSE simultaneously, then export them for the existing split reduction. |

The timing experiments price the whole schedule. A saved live range can cost
extra loads, synchronization, or registers. On DGX Spark, long-prefill
candidates reached **1.11–1.13×** against default FA2 over 13 shapes at
8K–32K. The same-template comparison shows a much smaller contribution from
the shared-memory handoff itself. The full results include short-shape and
long-decode regressions. [Measurements](docs/measurements.md)

## Install and solve on CPU

Use Python 3.10 or newer. GPU execution uses CUDA-enabled PyTorch, a compatible
CUDA Toolkit with `nvcc`, a C++20 compiler, and Ninja on PATH.

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m tilecadence.exchange --out runs/exchange
python -m tilecadence.operator --phase prefill --out runs/prefill
python -m tilecadence.operator --phase decode --out runs/decode
```

`exchange` emits the ordered operations, placements, dependency replay, and
CUDA source for each arm. `operator` emits the long-operator template with
the offsets from the placement solver.

## Build and measure on a CUDA GPU

Builds target the selected NVIDIA device at runtime, starting at compute
capability 8.0. The installed toolkit must support the selected target.
The two-tile and fused decode schedules use 32 KiB of shared memory per
CTA. The grouped prefill schedule requires 96 KiB of opt-in shared memory.
The two-tile demo is FP16/BF16 MHA with sequence length 64 and head dimension
128. The long operators are BF16 Qwen-style GQA with 32 query heads, 8 KV
heads, head dimension 128, batch 1 or 4, and 128–32768 tokens. The prefill
baseline requires the upstream heuristic to use one split for the
chosen device and shape.

```bash
git clone https://github.com/Dao-AILab/flash-attention.git ../flash-attention
git -C ../flash-attention checkout 49774b31c8d1172a5b28f72e9aed0f322bf4ccba
git -C ../flash-attention submodule update --init csrc/cutlass
python -m pip install ninja

python -m tilecadence.bench_exchange --flash-attention-root ../flash-attention \
  --plans runs/exchange --work runs/exchange-check --verify-only
python -m tilecadence.bench_operator --flash-attention-root ../flash-attention \
  --phase prefill --source runs/prefill --batches 1 --lengths 128 2048 8192 \
  --work runs/prefill-bench
python -m tilecadence.bench_operator --flash-attention-root ../flash-attention \
  --phase decode --source runs/decode --batches 1 --lengths 128 2048 8192 \
  --work runs/decode-bench
```

Pass `--device cuda:N` to select a visible device. The operator benchmark
uses seeded synthetic inputs, checks O and LSE against the pinned original
FA2 CUDA dispatch, and times preallocated calls with alternating GPU event
pairs. Decode checks the previous/current results and both cache appends.
Use `--verify-only` for a correctness smoke test. Each run writes its build
identity, measured pairs, and compiled resources to a fresh directory.
The two-tile benchmark also supports timed comparisons against an installed
CUDA FA2 package and records that package's identity separately.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the FA2 and CUTLASS
revisions and license terms.
