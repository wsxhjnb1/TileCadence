# FA2 Residency Scheduler

Constraint-guided CUDA/CuTe scheduling for FlashAttention-2. The solver explores where to keep Q, when shared K storage can be reused, how to overlap the next K load with work on the current tile, and how to split KV work during decode. Measured device time and compiler resource usage guide the next search round.

```text
B · solve legal tile, lifetime, and copy/wait schedules
       ↓ emit CUDA/CuTe and compile
A · fit candidate cost from latency, registers, shared memory, and occupancy
       ↓ propose unseen schedules for B to solve
     lock a per-shape policy and validate inside Qwen3
```

## DGX Spark results

Qwen3-4B, all 36 layers, BF16, GB10 (SM121). Baseline is the **native default** path from pinned upstream FlashAttention-2. Speedup is native time divided by scheduled time.

| Batch | Prompt | Prefill attention | Prefill forward | Decode attention | Decode forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8,192 | 1.0313× | 1.0032× | 1.0408× | 0.9990× |
| 1 | 32,768 | 1.0498× | 1.0139× | 1.0040× | 1.0053× |
| 4 | 8,192 | 1.0587× | 1.0074× | 1.0160× | 1.0060× |
| 4 | 32,768 | **1.0397×** | **1.0161×** | 1.0022× | 1.0000× |

At batch 4 and 32K, full prefill went from **36,630.5 to 36,050.4 ms** (95% process-cluster bootstrap interval for speedup 1.0152×–1.0170×). Attention is attributed to the main and required combine kernels in separate CUPTI traces. Full forward uses unprofiled CUDA-event intervals. Candidates were selected on development inputs before model confirmation.

All 90 fixed-history next-token choices matched. Eight full attention O/LSE comparisons passed, with maximum absolute differences of 0.001953125 (O) and 1.04904e-5 (LSE). The separate later batch-4/32K **fused KV append** decode study found essentially flat attention (79.743 → 79.700 ms) and forward latency (122.872 → 123.085 ms), so it provides no stable decode speedup. The table uses the original growing DynamicCache path.

[Timing data](evidence/model_timing.json) · [Numerical checks](evidence/numerical_proof.json) · [Measurement notes](evidence/README.md)

## Generate schedules on a CPU

Use Python 3.11 or 3.12. All commands run from the repository root. On Windows, clone into a short directory path to accommodate the vendored CUTLASS headers.

```bash
python -m pip install -r requirements.txt
mkdir -p work
python solver_attention/spark/qwen3_long_ab_b_stage_v14.py catalog --out work/catalog.json
python solver_attention/spark/qwen3_long_ab_controller_v14.py plan \
  --catalog work/catalog.json --out work/round0.json
python solver_attention/spark/qwen3_long_ab_b_stage_v14.py emit \
  --catalog work/catalog.json --plan work/round0.json --out work/emitted
```

The generated source directories include their solved schedules and file hashes. The retained first-round measurements can drive the A feedback step on a CPU. See [feedback replay](evidence/README.md#replay-hardware-feedback-on-a-cpu).

## Compare the prefill operator on a GPU

Use Linux, a CUDA 13.0 PyTorch environment, a CUDA 13.0 toolkit, a C++20 host compiler, Ninja, and an SM121 GPU. The adapter builds the selected candidate and the upstream default path, checks complete O and LSE, and alternates role order for paired timings. This smoke benchmark uses synthetic strided Q/K/V at the Qwen3 shape, so it measures a single attention operator. The model table above comes from the retained pretrained-model study.

```bash
python -m pip install -r requirements.txt ninja
CUDA_HOME=/usr/local/cuda-13.0 python bench_prefill.py --batch 4 --length 32768 --work work/prefill
```

Scope is dense causal prefill and single-query decode with Qwen3-4B head geometry (32 query heads, 8 KV heads, head dimension 128). The public build entry point covers prefill. The solver also emits split-KV decode source. The original model checkpoint, captured tensors, and compiled binaries are kept out of the source distribution.

## License

Project code is MIT. Vendored FlashAttention and CUTLASS/CuTe code, and generated kernels derived from them, retain their upstream notices. See [third-party notices](THIRD_PARTY_NOTICES.md).
