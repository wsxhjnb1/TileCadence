# Measurement notes

The table in the repository README comes from the retained Qwen3-4B long-context A+B study. The baseline calls the default upstream dispatch at FlashAttention commit `49774b31c8d1172a5b28f72e9aed0f322bf4ccba`. The model revision is `1cfa9a7208912126459214e8b04321603b3df60c`.

- **Development.** Two rounds of four prefill and four decode candidate requests each. Four cases per phase and three balanced paired CUDA-event observations per case. All 64 cells passed complete attention O/LSE checks. Only development results were used to select the policies.
- **Model confirmation.** Four fresh processes, one complete warmup per role and case, then two paired rounds. The 576 clean rows include eight growing decode steps per trajectory using fixed native-produced token histories. The bootstrap resamples whole processes (10,000 draws).
- **Attention attribution.** Separate CUPTI processes retained 16 traces covering 48 forwards and 1,728 attention calls. The attention metric sums attributed main and required combine kernels across all layers. Attention process ratios are descriptive.
- **Numerics.** Eight layer-0 captures were checked in full for O/LSE. The independent FP32 reference checks sampled query rows with all keys, heads and batches. The fixed-history top-1 comparisons cover all 90 choices.
- **Acquisition audit.** A CPU replay replaced an overly strict fixed CUPTI/CUDA-event span check with the synchronized host bound, preserving every measured duration and validating trace ownership and bounds.

[Model timing summary](model_timing.json) is the retained analysis export with local paths and the device identifier redacted. [Numerical proof](numerical_proof.json) retains the original record. Full tensors and raw profiler traces remain in the original study archive.

## Replay hardware feedback on a CPU

`development/catalog.json` is the original catalog. `development/r0_plan.json` is the exact plan used for measurement after a native-receipt field fix. The candidate selection and compiled inputs stayed the same through that fix. `development/r0_evaluation.json` preserves all 32 measured cells and their 96 timing pairs, along with artifact hashes. Only the local path prefixes in build receipts and B proof references were redacted.

```bash
mkdir -p work
python solver_attention/spark/qwen3_long_ab_controller_v14.py feedback \
  --catalog evidence/development/catalog.json \
  --plan evidence/development/r0_plan.json \
  --evaluation evidence/development/r0_evaluation.json \
  --out work/feedback.json
```

The replay fits the cost model on measured CUDA-event latency and compiler resources. It reproduces the four prefill and four decode proposals, in order, from the retained second round. The new plan records the hash of the redacted evaluation. Use the fresh catalog and emission commands in the top-level README to generate source with the current B-stage script. The historical catalog binds the earlier B-stage source.
