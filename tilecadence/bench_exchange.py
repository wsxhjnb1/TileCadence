"""Check and time the generated two-tile attention residency schedules."""

import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
from statistics import geometric_mean, median

from .cuda_runtime import FA2_COMMIT, build_cuda, cuda_device, file_sha256


EXCHANGE_ARMS = ("one", "multi", "relocated_two_p")
RESOURCE_KEYS = (
    "block_m", "block_n", "head_dim", "value_stripe_dim", "threads",
    "dynamic_shared_bytes", "static_shared_bytes", "registers_per_thread",
    "local_bytes_per_thread", "binary_version", "ptx_version",
    "theoretical_active_ctas_per_sm", "required_nonce_elements", "diagnostic_guard",
)
GRAPH_CALLS = 16
REPLAYS = 32


def _stock_forward(causal):
    try:
        stock = import_module("flash_attn_2_cuda")
        distribution = metadata.distribution("flash-attn")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Timing requires installed CUDA FlashAttention-2; use --verify-only for correctness"
        ) from error
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")

    def forward(slot):
        q, k, v, out, _ = slot
        actual_out, lse, _, _ = stock.fwd(
            q, k, v, out, None, 0.0, 128 ** -0.5, causal, -1, -1, 0.0, False, None
        )
        if actual_out.data_ptr() != out.data_ptr():
            raise RuntimeError("Stock FlashAttention did not use the supplied output buffer")
        return actual_out, lse

    return forward, {
        "implementation": "installed upstream flash_attn_2_cuda.fwd, default dispatch",
        "version": distribution.version,
        "installed_vcs_commit": installed_commit,
        "matches_header_revision_in_install_metadata": installed_commit == FA2_COMMIT,
        "extension_sha256": file_sha256(stock.__file__),
        "preallocated_output": True,
        "preallocated_lse": False,
    }


def _reference(torch, slot, causal):
    q, k, v = (tensor.transpose(1, 2).float() for tensor in slot[:3])
    scores = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
    if causal:
        index = torch.arange(q.size(2), device=q.device)
        scores.masked_fill_(index[None, :] > index[:, None], float("-inf"))
    return (scores.softmax(dim=-1) @ v).transpose(1, 2), scores.logsumexp(dim=-1)


def _assert_close(torch, actual, expected, dtype):
    atol, rtol = (0.003, 0.01) if dtype == torch.float16 else (0.02, 0.02)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(actual[1], expected[1], atol=0.0005, rtol=0.0001)


def _candidate_forward(torch, ops, query, causal):
    raw = ops.resources(query, causal)
    resources = dict(zip(RESOURCE_KEYS, raw, strict=True))
    if resources["theoretical_active_ctas_per_sm"] < 1 or resources["diagnostic_guard"]:
        raise RuntimeError(f"Generated kernel cannot be used for benchmarking: {resources}")
    nonce = torch.zeros(resources["required_nonce_elements"], device=query.device, dtype=torch.int32)

    def forward(slot):
        q, k, v, out, lse = slot
        ops.forward_out(q, k, v, out, lse, nonce, causal, True, False)
        return out, lse

    return forward, resources


def _make_slots(torch, args, dtype, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch, 64, args.heads, 128)
    slots = []
    for _ in range(args.slots):
        qkv = tuple(torch.randn(shape, dtype=dtype, device=device, generator=generator)
                    for _ in range(3))
        slots.append((*qkv, torch.empty_like(qkv[0]),
                      torch.empty((args.batch, args.heads, 64), device=device, dtype=torch.float32)))
    return slots


def _validate(torch, arms, slots, causal, dtype):
    for slot in slots:
        reference = _reference(torch, slot, causal)
        retain = tuple(tensor.clone() for tensor in arms["retain"](slot))
        _assert_close(torch, retain, reference, dtype)
        baseline = arms.get("official_default")
        stock = tuple(tensor.clone() for tensor in baseline(slot)) if baseline else None
        if stock is not None:
            _assert_close(torch, stock, reference, dtype)
        for name, forward in arms.items():
            if name in ("retain", "official_default"):
                continue
            actual = forward(slot)
            if not all(torch.equal(a.view(integer), b.view(integer))
                       for a, b, integer in zip(actual, retain, (torch.int16, torch.int32), strict=True)):
                raise RuntimeError(f"{name} differs bitwise from the retain schedule")
            if stock is not None:
                _assert_close(torch, actual, stock, dtype)


def _graph(torch, forward, slots):
    forward(slots[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(max(GRAPH_CALLS, len(slots))):
            forward(slots[index % len(slots)])
    return graph


def _time_graph(torch, graph, calls):
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * calls)


def _benchmark(torch, arms, slots, rounds):
    timings = {}
    names = tuple(arms)
    for regime, buffers in (("repeated", slots[:1]), ("rotating", slots)):
        graphs = {name: _graph(torch, arms[name], buffers) for name in names}
        samples = {name: [] for name in names}
        for index in range(rounds):
            offset = index % len(names)
            order = names[offset:] + names[:offset]
            if index % 2:
                order = order[::-1]
            for name in order:
                samples[name].append(_time_graph(torch, graphs[name], max(GRAPH_CALLS, len(buffers))))
        timings[regime] = {
            "us_per_call": samples,
            "median_us_per_call": {name: median(values) for name, values in samples.items()},
            "paired_speedup_vs_official_default": {
                name: geometric_mean(
                    baseline / value for baseline, value in
                    zip(samples["official_default"], samples[name], strict=True)
                )
                for name in names if name != "official_default"
            },
        }
    return timings


def _emitted_source(args, role):
    if args.plans is None:
        from .emit_exchange import emit
        from .exchange import build_problem
        from .model import solve

        source = args.work / role / "emitted"
        source.parent.mkdir()
        problem = build_problem(role)
        result = solve(problem, max_time_seconds=15, artifact_dir=source.parent / "solver")
        emit(problem, result, source, opaque_recovery=False)
    else:
        source = args.plans / role / "emitted"

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = {**manifest["backend_source_sha256"], "plan.cu": manifest["cuda_source_sha256"]}
    if set(hashes) != {"plan.cu", "backend.cuh", "binding.cpp", "output_bits.cuh"}:
        raise ValueError(f"Unexpected generated files for {role}")
    source_files = {path.name for path in source.iterdir()
                    if path.suffix in (".cpp", ".cu", ".cuh", ".h", ".hpp")}
    if source_files != set(hashes):
        raise ValueError(f"Generated file list changed for {role}")
    for filename, expected in hashes.items():
        if file_sha256(source / filename) != expected:
            raise ValueError(f"Generated source hash mismatch for {role}/{filename}")
    return source


def run(args):
    import torch

    device, capability = cuda_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    args.work.mkdir(parents=True, exist_ok=False)
    roles = ("retain", *(args.arm or EXCHANGE_ARMS))
    receipt = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": capability,
        "shape": [args.batch, 64, args.heads, 128],
        "dtype": args.dtype,
        "causal": args.causal,
        "seed": args.seed,
        "slots": args.slots,
        "candidates": {},
    }
    with torch.cuda.device(device), torch.inference_mode():
        arms = {}
        if not args.verify_only:
            arms["official_default"], receipt["baseline"] = _stock_forward(args.causal)
        slots = _make_slots(torch, args, dtype, device)
        for role in roles:
            print(f"Preparing and compiling {role} for {receipt['device']}", flush=True)
            source = _emitted_source(args, role)
            ops, build = build_cuda(source, args.flash_attention_root, device)
            arms[role], resources = _candidate_forward(torch, ops, slots[0][0], args.causal)
            receipt["candidates"][role] = {"build": build, "resources": resources}
        _validate(torch, arms, slots, args.causal, dtype)
        receipt["correctness"] = {
            "all_slots_match_fp32_reference_with_tolerance": True,
            "exchange_output_and_lse_match_retain_bitwise": True,
            "exchange_matches_stock_with_tolerance": True if "official_default" in arms else None,
        }
        if not args.verify_only:
            receipt["timings"] = _benchmark(torch, arms, slots, args.rounds)
    (args.work / "result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.work / 'result.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-attention-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--plans", type=Path, help="Reuse source generated by tilecadence.exchange")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=192)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7819)
    parser.add_argument("--arm", choices=EXCHANGE_ARMS, action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.heads, args.slots, args.rounds) < 1:
        parser.error("Batch, heads, slots, and rounds must be positive")
    if args.arm and len(args.arm) != len(set(args.arm)):
        parser.error("Each exchange schedule can be selected once")
    run(args)


if __name__ == "__main__":
    main()
