"""Check and time the generated Qwen-style prefill or fused decode operator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import geometric_mean
from time import perf_counter_ns

from .cuda_runtime import build_cuda, cuda_device


def workspace(torch, spec, device):
    return tuple(torch.empty(size, device=device, dtype=torch.float32) for size in spec[1:3])


def equal_bits(torch, expected, actual):
    for want, got, integer in zip(expected, actual, (torch.int16, torch.int32), strict=True):
        if not torch.isfinite(got).all() or not torch.equal(want.view(integer), got.view(integer)):
            raise RuntimeError("O or LSE differs from the original upstream dispatch")


def resource_summary(raw):
    keys = ("threads", "dynamic_shared_bytes", "static_shared_bytes", "registers_per_thread",
            "local_bytes_per_thread", "binary_version", "ptx_version", "active_ctas_per_sm")
    return dict(zip(keys, raw[12:20], strict=True))


def prefill(torch, ops, sample, batch, length):
    q, k, v = (sample(batch, length, h, 128) for h in (32, 8, 8))
    arms = []
    for mapping in (0, 1):
        out = torch.empty_like(q)
        lse = torch.empty((batch, 32, length), device=q.device, dtype=torch.float32)
        ws = workspace(torch, ops.workspace_spec(q, k, v, 0, mapping, True), q.device)

        def invoke(out=out, lse=lse, ws=ws, mapping=mapping):
            ops.forward_out(q, k, v, out, lse, *ws, 0, mapping, 128 ** -0.5, True)

        invoke()
        arms.append((invoke, lambda: None, (out, lse)))
    equal_bits(torch, arms[0][2], arms[1][2])
    return arms, resource_summary(ops.resources(q, k, v, 0, 1, True))


def decode(torch, ops, sample, batch, length):
    prefix = tuple(sample(batch, length, 8, 128) for _ in range(2))
    steps = [tuple(sample(batch, 1, h, 128) for h in (32, 8, 8)) for _ in range(2)]
    device = prefix[0].device
    lengths = [torch.full((batch,), length + step, dtype=torch.int32, device=device) for step in range(2)]
    arms, previous_results = [], []
    for mapping in (0, 1):
        cache = tuple(torch.empty((batch, length + 2, 8, 128), dtype=p.dtype, device=device) for p in prefix)
        for c, p in zip(cache, prefix, strict=True):
            c[:, :length].copy_(p)
        calls, outputs = [], []
        for step, (q, k, v) in enumerate(steps):
            out = torch.empty_like(q)
            lse = torch.empty((batch, 32, 1), dtype=torch.float32, device=device)
            ws = workspace(torch, ops.workspace_spec(q, cache[0], lengths[step], 0, mapping, True), device)

            def invoke(q=q, k=k, v=v, n=lengths[step], out=out, lse=lse, ws=ws,
                       cache=cache, mapping=mapping):
                ops.forward_out(q, *cache, k, v, n, out, lse, *ws, 0, mapping, True)

            invoke()
            calls.append(invoke)
            outputs.append((out, lse))
        for idx, (c, p) in enumerate(zip(cache, prefix, strict=True)):
            for want, got in [(p, c[:, :length]), (steps[0][idx + 1], c[:, length:length + 1]),
                              (steps[1][idx + 1], c[:, length + 1:length + 2])]:
                if not torch.equal(want.view(torch.int16), got.view(torch.int16)):
                    raise RuntimeError("Fused append changed historical or newly appended KV")
        arms.append((calls[1], calls[0], outputs[1]))
        previous_results.append(outputs[0])
    equal_bits(torch, *previous_results)
    equal_bits(torch, arms[0][2], arms[1][2])
    return arms, resource_summary(ops.resources(steps[1][0], cache[0], lengths[1], 0, 1, True))


def timed(torch, arm):
    invoke, warm_previous, _ = arm
    lead, start, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    warm_previous()
    lead.record()
    queued_at = perf_counter_ns()
    torch.cuda._sleep(2_000_000)
    start.record()
    invoke()
    end.record()
    enqueue_ns = perf_counter_ns() - queued_at
    end.synchronize()
    if lead.elapsed_time(start) * 1e6 <= enqueue_ns:
        raise RuntimeError("The GPU reached the timed region before the host finished enqueueing")
    return start.elapsed_time(end) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--source", type=Path, help="Use a directory previously emitted by tilecadence.operator")
    parser.add_argument("--flash-attention-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, choices=(1, 4), nargs="+", default=(1, 4))
    parser.add_argument("--lengths", type=int, nargs="+", default=(128, 2048, 8192))
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not all(128 <= n <= 32768 for n in args.lengths) or args.rounds < 2:
        parser.error("Lengths must be in [128, 32768] and rounds must be at least two")

    import torch

    args.work.mkdir(parents=True, exist_ok=False)
    source = args.source or args.work / "source"
    if args.source is None:
        from .operator import emit

        emit(args.phase, source)
    device, _ = cuda_device(args.device)
    with torch.cuda.device(device), torch.inference_mode():
        ops, build = build_cuda(source, args.flash_attention_root, device)
        generator = torch.Generator(device=device).manual_seed(7911)

        def sample(*shape):
            return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)

        run = prefill if args.phase == "prefill" else decode
        with (args.work / "rows.jsonl").open("x", encoding="utf-8") as raw:
            for batch in args.batches:
                for length in args.lengths:
                    arms, resources = run(torch, ops, sample, batch, length)
                    row = dict(phase=args.phase, batch=batch, length=length, bitwise=True, resources=resources)
                    if not args.verify_only:
                        for _ in range(4):
                            for invoke, warm_previous, _ in arms:
                                warm_previous()
                                invoke()
                        torch.cuda.synchronize(device)
                        windows = []
                        for rnd in range(args.rounds):
                            pair = [0.0, 0.0]
                            for role in ((0, 1) if rnd % 2 == 0 else (1, 0)):
                                pair[role] = timed(torch, arms[role])
                            windows.append(dict(default_us=pair[0], candidate_us=pair[1]))
                        row.update(windows=windows, speedup=geometric_mean(
                            p["default_us"] / p["candidate_us"] for p in windows))
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()
                    print(json.dumps(row), flush=True)
    (args.work / "build.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
