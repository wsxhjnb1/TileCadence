# Measurements

## Whole attention operators on DGX Spark

The data uses real BF16 Qwen3-4B layer-zero projections, batch 1 or 4,
and 30 equal-length shapes from 128 through 32768 tokens. The denominator
is the default CUDA dispatch from the pinned FA2 revision. Prefill is
causal. Decode is warmed with a preceding Q and fused append. The timed
decode includes the current Q, its fused KV append, and split reduction.
O, LSE, and the two decode append results pass bitwise checks on all
30 shapes for both published schedules.

| Candidate | Geometric mean, all 30 shapes | Range | Confirmed wins |
|---|---:|---:|---:|
| Prefill with vector LSE writeback | 1.0611× | 0.6215×–1.1345× | 25/30 |
| Decode with O and LSE sharing the terminal V range | 1.0636× | 0.9769×–1.4032× | 11/30 |

The [per-shape CSV](spark_speedups.csv) preserves both confirmation ratios
for every shape. A confirmed win requires both ratios above 1 and their
geometric mean at least 1.005. The operator study used CUPTI through 2048
tokens and guarded CUDA events above that. Each process collected five
alternating paired blocks, with three samples per arm.

The prefill route at 8K–32K reaches 1.1121×–1.1345× on all 13 shapes.
Across 21 shapes shared by the scalar handoff and the original prefill
template, development ratios were 1.1095× and 1.1082×, respectively.
For decode, terminal O/LSE reuse and the same template retaining the
original epilogue gave 1.0474× and 1.0613× over 30 development shapes.
The native-relative gains include the existing tiling and pipeline.

## Two probability checkpoints

The two-tile MHA experiment used 64 tokens, head dimension 128, and 192
heads. The two-checkpoint variant stores both P tiles in separate 4 KiB
spans of the original Q region. Reloaded Q feeds the second QK from a
different address, and the two P loads feed their corresponding PVs.
The P/V and two-P variants were bitwise equal to the same-math retain arm
in 128/128 input, dtype, and mask checks per arm. Comparisons against
default FA2 passed FP16/BF16 O and FP32 LSE tolerances. On rotating
buffers, the single-P variant averaged 1.021× in confirmation. On reused
buffers the retain arm was faster than the exchange variants.

The repository benchmark uses seeded synthetic tensors. The Spark
operator CSV uses saved real-model projections from the experiment above.

## Portable build checks

The published CUDA paths compiled and ran on a GeForce RTX 3080 Laptop
GPU (SM86, WSL) and NVIDIA GB10 (SM121, DGX Spark). On each GPU, the
prefill and fused decode benchmarks passed bitwise O/LSE checks at
batch 1 and lengths 128, 193, and 2048. Both devices also passed the
two-tile BF16 retain, P/V, and two-P checks against the FP32 reference
and the retain output. Spark additionally passed the FP16 unmasked
two-tile check. The performance table above reports the DGX Spark study.
