"""Build the selected FA2 prefill schedule and compare it with native default on SM121."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from solver_attention.long_prefill_b_v14 import UPSTREAM, VENDOR, VENDOR_SHA256, emit
from solver_attention.native_model_v11.prepare import RESOURCE_KEYS

BASELINE = Path(__file__).resolve().parent / "solver_attention/native_model_v11/emitted_v3/prefill_natural64"
PLANS = {
    (1, 8192): ("pf_m128n128w8_reload_forward_early", 404),
    (4, 8192): ("pf_m128n32w4_reload_reverse_early", 415),
    (1, 32768): ("pf_m128n128w8_alias_reverse_early", 402),
    (4, 32768): ("pf_m128n128w8_alias_reverse_early", 402),
}


def verify_vendor():
    manifest_path = VENDOR / "manifest.json"
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != VENDOR_SHA256:
        raise RuntimeError("Pinned vendor manifest changed")
    manifest = json.loads(data)
    if manifest["commit"] != UPSTREAM:
        raise RuntimeError("Wrong FlashAttention revision")
    for name, record in manifest["files"].items():
        if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Pinned vendor source changed: {name}")


def build(source, directory, name):
    from torch.utils.cpp_extension import load

    directory.mkdir()
    include_dirs = (
        source, VENDOR / "csrc/flash_attn", VENDOR / "csrc/flash_attn/src", VENDOR / "csrc/cutlass/include"
    )
    disabled = ("BACKWARD", "DROPOUT", "ALIBI", "SOFTCAP", "UNEVEN_K", "LOCAL")
    flags = ["-O3", "-std=c++20", "-DFLASH_NAMESPACE=TORCH_EXTENSION_NAME",
             *[f"-DFLASHATTENTION_DISABLE_{feature}" for feature in disabled]]
    cuda_flags = [
        *flags, "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--use_fast_math", "-lineinfo", "-Xptxas=-v", "--threads", "2",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    load(
        name=name,
        sources=[str(source / "binding.cpp"), str(source / "prefill.cu")],
        extra_include_paths=[str(path) for path in include_dirs],
        extra_cflags=flags,
        extra_cuda_cflags=cuda_flags,
        build_directory=str(directory),
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )


def benchmark(torch, batch, length, repetitions):
    generator = torch.Generator(device="cuda").manual_seed(17471)
    options = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn((batch, length, 32, 128), generator=generator, **options)
    k, v = [torch.randn((batch, 8, length, 128), generator=generator, **options).transpose(1, 2)
            for _ in range(2)]
    tensors = (q, k, v)
    scale = 1 / math.sqrt(128)
    arms = {
        "native": (torch.ops.fa2_residency_native, 0),
        "candidate": (torch.ops.fa2_residency_candidate, 1),
    }
    outputs = {}
    resources = {}
    for role, (ops, mapping) in arms.items():
        _, output_workspace, lse_workspace = ops.workspace_spec(*tensors, 0, mapping, True)
        outputs[role] = (
            torch.empty_like(q),
            torch.empty((batch, 32, length), device="cuda", dtype=torch.float32),
            torch.empty(output_workspace, device="cuda", dtype=torch.float32),
            torch.empty(lse_workspace, device="cuda", dtype=torch.float32),
        )
        resources[role] = dict(zip(RESOURCE_KEYS, ops.resources(*tensors, 0, mapping, True), strict=True))

    def forward(role):
        ops, mapping = arms[role]
        ops.forward_out(*tensors, *outputs[role], 0, mapping, scale, True)

    for role in arms:
        forward(role)
    torch.cuda.synchronize()

    native_output, native_lse = outputs["native"][:2]
    candidate_output, candidate_lse = outputs["candidate"][:2]
    torch.testing.assert_close(candidate_output, native_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(candidate_lse, native_lse, rtol=0.0001, atol=0.0005)

    pairs = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for pair in range(repetitions):
        order = tuple(arms) if pair % 2 == 0 else tuple(reversed(arms))
        elapsed = {}
        for role in order:
            start.record()
            forward(role)
            stop.record()
            stop.synchronize()
            elapsed[role] = start.elapsed_time(stop)
        pairs.append(elapsed)
    return {
        "workload": "single causal prefill attention operator with fixed synthetic strided Q/K/V",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "batch": batch,
        "length": length,
        "full_o_lse_comparison": "passed",
        "pairs_ms": pairs,
        "speedup": statistics.geometric_mean(row["native"] / row["candidate"] for row in pairs),
        "resources": resources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--length", type=int, choices=(8192, 32768), default=8192)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--work", type=Path, required=True, help="New directory for source and build outputs")
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error("--pairs must be an even integer >= 2")
    if args.work.exists():
        parser.error("--work must not exist")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
    os.environ.setdefault("MAX_JOBS", "1")
    import torch

    if torch.version.cuda != "13.0" or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        parser.error("This benchmark requires a CUDA 13.0 PyTorch environment and an SM121 CUDA GPU")
    verify_vendor()
    work = args.work.resolve()
    source = work / "candidate"
    plan, code = PLANS[(args.batch, args.length)]
    emit(source, plan, family_code=code)
    build(BASELINE, work / "native-build", "fa2_residency_native")
    build(source, work / "candidate-build", "fa2_residency_candidate")
    result = benchmark(torch, args.batch, args.length, args.pairs)
    result["candidate"] = plan
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
