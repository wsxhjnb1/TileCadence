"""Build and measure frozen Qwen3 A+B development candidates on Spark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import qwen3_long_ab_controller_v14 as policy


BUILD_SCHEMA = "qwen3-long-ab-build-batch-v14"
BASE_FAMILIES = {"prefill": "prefill_natural64", "decode": "decode_v6"}


def within(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    policy.require(not path.is_absolute() and ".." not in path.parts, "Expected bounded relative evidence path")
    path = (root / path).resolve()
    policy.require(path.is_relative_to(root), "Evidence path escapes its frozen artifact root")
    return path


def append(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def verify_receipt(path, expected_sources=None):
    receipt = policy.read(path)
    policy.require(receipt["state"] == "complete" and receipt["target"] == "sm121"
                   and receipt["vendor_manifest_sha256"] == policy.VENDOR_MANIFEST_SHA256,
                   f"Compiled Spark receipt is invalid: {path}")
    sources = receipt["source_sha256"]
    policy.require(sources and all(policy.sha(within(path.parent / "source", name)) == digest
                                  for name, digest in sources.items()), "Compiled source snapshots changed")
    if expected_sources is not None:
        policy.require(set(sources) == set(expected_sources) | {"build_extension.py"}
                       and all(sources.get(name) == digest for name, digest in expected_sources.items()),
                       "Build differs from B-emitted source bytes")
    module = Path(receipt["module_path"]).resolve()
    policy.require(module.is_relative_to(path.parent.resolve() / "build") and policy.sha(module) == receipt["module_sha256"],
                   "Compiled module path/hash mismatch")
    return dict(build_receipt=str(path.resolve()), build_receipt_sha256=policy.sha(path),
                module_sha256=receipt["module_sha256"], source_sha256=sources)


def build(args, plan, trials):
    emission = policy.read(args.emission)
    policy.require(emission["schema"] == "qwen3-long-ab-b-emission-v14" and emission["state"] == "solved_emitted"
                   and emission["plan_sha256"] == policy.sha(args.plan) and emission["round_id"] == plan["round_id"]
                   and emission["catalog_sha256"] == plan["catalog_sha256"],
                   "B emission belongs to another exact A plan")
    entries = {entry["candidate_id"]: entry for entry in emission["candidates"]}
    policy.require(len(entries) == len(emission["candidates"]) == len(trials)
                   and set(entries) == {row["candidate_id"] for row in trials}, "Every trial needs one emission outcome")
    root = args.workspace.resolve()
    helper = root / "src/build_extension_v6.py"
    policy.require(helper.is_file(), "Pinned Spark CUDA extension builder is unavailable")
    policy.require(re.fullmatch(r"[a-z][a-z0-9_]{0,30}", args.run_prefix), "Fresh bounded build prefix required")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                   text=True, timeout=20)
    policy.require(not apps.strip(), "Another compute process is active")
    jobs = {}
    bindings = {}
    for trial in trials:
        id_ = trial["candidate_id"]
        entry = entries[id_]
        sources = entry["source_sha256"]
        directory = within(args.emission.parent, entry["source_dir"])
        proof = within(args.emission.parent, entry["b_plan"])
        policy.require(policy.sha(proof) == entry["b_plan_sha256"], f"B solve evidence changed for {id_}")
        solved = policy.read(proof)
        policy.require(solved["schema"] == "qwen3-long-ab-b-proof-v14" and solved["state"] == "solved_emitted"
                       and all(solved[key] == trial[key] for key in ("candidate_id", "phase", "parameters", "controls"))
                       and solved["source_sha256"] == sources and solved["solved"],
                       "B solve must bind the exact chosen request and emitted source")
        policy.require(set(sources) == {p.name for p in directory.iterdir() if p.is_file()}
                       and all(Path(name).name == name and policy.sha(directory / name) == digest
                               for name, digest in sources.items()), f"B-emitted sources changed for {id_}")
        policy.require("binding.cpp" in sources and any(name.endswith(".cu") for name in sources),
                       "Both CUDA and binding sources are required")
        source_key = json.dumps(sources, sort_keys=True)
        if source_key not in jobs:
            run_id = f"{args.run_prefix}_{id_}"
            policy.require(not (root / "results" / run_id).exists(), f"Keep existing build {run_id}")
            jobs[source_key] = (run_id, directory, sources)
        bindings[id_] = (source_key, entry["b_plan_sha256"], str(proof))
    args.out.mkdir(parents=True, exist_ok=False)
    for path, name in ((args.plan, "plan.json"), (args.emission, "emission.json"), (helper, "build_extension_v6.py")):
        (args.out / name).write_bytes(path.read_bytes())
    policy.write_new(args.out / "build_slot.json", {"gpu_compute": apps, "pid": os.getpid(), "lanes": 2})

    def compile_one(job):
        run_id, directory, sources = job
        log = args.out / f"{run_id}.log"
        command = [sys.executable, str(helper), "--workspace", str(root), "--source", str(directory), "--run-id", run_id]
        start = time.perf_counter()
        with log.open("xb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
        receipt = root / "results" / run_id / "receipt.json"
        row = dict(run_id=run_id, state="complete" if result.returncode == 0 else "compile_failed",
                   exit_code=result.returncode, wall_seconds=time.perf_counter() - start,
                   builder_stdout_sha256=policy.sha(log), emitted_source_sha256=sources)
        if result.returncode == 0:
            row.update(verify_receipt(receipt, sources))
        elif receipt.exists():
            row.update(build_receipt=str(receipt), build_receipt_sha256=policy.sha(receipt))
        policy.write_new(args.out / f"{run_id}.json", row)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(zip(jobs, pool.map(compile_one, jobs.values())))
    rows = []
    for id_, (key, proof_sha, proof) in bindings.items():
        rows.append(dict(candidate_id=id_, b_plan_sha256=proof_sha, b_plan=proof, **results[key]))
    policy.write_new(args.out / "builds.json", dict(schema=BUILD_SCHEMA, state="complete",
                     plan_sha256=policy.sha(args.plan), emission_sha256=policy.sha(args.emission),
                     round_id=plan["round_id"], rows=rows, unique_builds=len(jobs), lanes=2,
                     gpu_kernels_launched=False))


def load_captures(directory):
    index = directory / "validation_captures.jsonl"
    policy.require(policy.sha(index) == policy.CAPTURE_INDEX_SHA256, "Exact V13 actual capture index required")
    captures = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["case"] not in policy.CASES or row["chosen_role"] != "official_default":
            continue
        key = (row["case"], row["phase"])
        policy.require(key not in captures and row["passed"] and row["same_actual_QKV_for_both_kernels"]
                       and row["role"] == "official_default" and row["layer"] == 0,
                       "Expected unique actual native-path layer-zero input")
        batch, length = policy.CASE_DIMENSIONS[row["case"]]
        prefill = row["phase"] == "prefill"
        policy.require(row["phase"] in policy.PHASES and row["step"] == (-1 if prefill else 7)
                       and row["causal"] is prefill and math.isclose(row["scale"], 128 ** -.5, rel_tol=1e-12),
                       "Wrong captured prefill or grown decode semantics")
        expected = dict(q=[batch, length if prefill else 1, 32, 128],
                        k=[batch, length if prefill else length + 8, 8, 128],
                        v=[batch, length if prefill else length + 8, 8, 128])
        policy.require(all(row[name]["shape"] == shape and row[name]["dtype"] == "torch.bfloat16"
                           and row[name]["stride"][-1] == 1 for name, shape in expected.items()),
                       "Actual capture geometry differs from Qwen3")
        captures[key] = row
    policy.require(set(captures) == {(case, phase) for case in policy.CASES for phase in policy.PHASES},
                   "All eight real DEV sites are required")
    return captures


def restore_inputs(directory, row):
    import torch

    path = within(directory, row["full_tensor_file"])
    policy.require(policy.sha(path) == row["full_tensor_file_sha256"], "Actual QKV payload changed")
    source = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    restored = []
    for name in ("q", "k", "v"):
        meta = row[name]
        tensor = source[name]
        policy.require(list(tensor.shape) == meta["shape"] and tensor.dtype == torch.bfloat16, "Wrong retained QKV payload")
        result = torch.empty_strided(meta["shape"], meta["stride"], device="cuda", dtype=tensor.dtype)
        result.copy_(tensor)
        policy.require(list(result.stride()) == meta["stride"], "Original captured stride was lost")
        restored.append(result)
    return restored


def prepare_call(ops, tensors, controls, capture, keys):
    import torch

    q, k, v = tensors
    tail = (controls["num_splits"], controls["mapping"])
    args = (q, k, v, *tail, capture["causal"])
    specification = list(ops.workspace_spec(*args))
    raw = list(ops.resources(*args))
    policy.require(len(raw) == len(keys) and len(specification) == 3, "Compiled V11 strided ABI changed")
    resources = dict(zip(keys, raw))
    policy.require(specification == [resources[name] for name in ("effective_splits", "workspace_o_float_elements",
                    "workspace_lse_float_elements")], "Launch and resource workspace disagree")
    if controls["num_splits"]:
        policy.require(specification[0] == controls["num_splits"], "Candidate did not execute its solved split count")
    policy.require(resources["query_length"] == q.shape[1] and resources["key_length"] == k.shape[1]
                   and resources["query_heads"] == 32 and resources["kv_heads"] == 8
                   and resources["main_binary_version"] == 121 and resources["main_theoretical_active_ctas_per_sm"] > 0
                   and resources["causal_effective"] == int(capture["causal"]), "Wrong actual compiled geometry")
    for name, tensor in zip(("q", "k", "v"), tensors):
        for kind, stride in zip(("batch", "row", "head"), tensor.stride()):
            policy.require(resources[f"{name}_{kind}_stride"] == stride, "Kernel did not consume the original stride")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
    workspace = [torch.empty(size, dtype=torch.float32, device=q.device) for size in specification[1:]]

    def invoke():
        ops.forward_out(q, k, v, output, lse, *workspace, *tail, capture["scale"], capture["causal"])

    return dict(invoke=invoke, output=output, lse=lse, resources=resources, workspace_spec=specification)


def check_all(candidate, native):
    import torch

    checks = {}
    for name, dim, atol, rtol in (("output", 1, .02, .02), ("lse", 2, .0005, .0001)):
        flags, maxima, sums = [], [], []
        left, right = candidate[name], native[name]
        policy.require(left.shape == right.shape, "O/LSE output shape differs")
        for start in range(0, left.shape[dim], 256):
            size = min(256, left.shape[dim] - start)
            a, b = left.narrow(dim, start, size).float(), right.narrow(dim, start, size).float()
            delta = (a - b).abs()
            flags.append((torch.isfinite(a) & torch.isfinite(b) & (delta <= atol + rtol * b.abs())).all())
            maxima.append(delta.max())
            sums.append(delta.sum(dtype=torch.float64))
        maximum = float(torch.stack(maxima).max())
        mean = float(torch.stack(sums).sum()) / left.numel()
        checks[name] = dict(ok=bool(torch.stack(flags).all()), elements=left.numel(), atol=atol, rtol=rtol,
                            max_abs=maximum if math.isfinite(maximum) else None,
                            mean_abs=mean if math.isfinite(mean) else None)
    checks["ok"] = checks["output"]["ok"] and checks["lse"]["ok"]
    return checks


def measure(args, plan, trials):
    import torch
    import runtime

    batch = policy.read(args.builds)
    policy.require(batch["schema"] == BUILD_SCHEMA and batch["state"] == "complete"
                   and batch["plan_sha256"] == policy.sha(args.plan) and batch["round_id"] == plan["round_id"],
                   "Completed build batch must match the exact DEV plan")
    builds = {row["candidate_id"]: row for row in batch["rows"]}
    policy.require(len(builds) == len(batch["rows"]) == len(trials)
                   and set(builds) == {trial["candidate_id"] for trial in trials}, "Missing or duplicate build result")
    policy.require(policy.sha(args.base_manifest) == policy.BASE_MANIFEST_SHA256, "Original native V11 binding required")
    base = policy.read(args.base_manifest)
    path = Path(base["native_mapping"]["path"])
    policy.require(policy.sha(path) == base["native_mapping"]["sha256"], "Original strided ABI mapping changed")
    keys = policy.read(path)["resource_keys"]
    policy.require(len(keys) == 46 and len(set(keys)) == 46, "Original 46-field strided ABI required")
    captures = load_captures(args.capture_dir)
    args.out.mkdir(parents=True, exist_ok=False)
    for path, name in ((args.plan, "plan.json"), (args.builds, "builds.json"),
                       (args.capture_dir / "validation_captures.jsonl", "validation_captures.jsonl"),
                       (args.base_manifest, "base_manifest.json")):
        (args.out / name).write_bytes(path.read_bytes())
    source = args.out / "source"
    source.mkdir()
    for name, digest in plan["source_pins"].items():
        (source / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        policy.require(policy.sha(source / name) == digest, "Executed DEV snapshot differs from plan")
    occupancy = args.out / "occupancy.jsonl"

    def observe(stage, **fields):
        obs = runtime.observation()
        append(occupancy, dict(stage=stage, **fields, **obs))
        runtime.require_slot(obs)

    observe("start")
    hardware = runtime.hardware()
    policy.write_new(args.out / "hardware.json", hardware)
    props = hardware["properties"]
    policy.require((props["major"], props["minor"], props["multi_processor_count"]) == (12, 1, 48)
                   and hardware["torch"] == base["runtime"]["torch"], "Pinned Spark SM121 and Torch required")
    loaded = {}

    def load(identity):
        path = Path(identity["build_receipt"])
        key = identity["module_sha256"]
        if key not in loaded:
            observed = verify_receipt(path, identity.get("emitted_source_sha256"))
            policy.require(all(observed[k] == identity[k] for k in ("build_receipt_sha256", "module_sha256")),
                           "Executable differs from its frozen build binding")
            loaded[key], _ = runtime.load_ops(path)
        return loaded[key]

    native_identities = {}
    for phase, family in BASE_FAMILIES.items():
        item = base["native_builds"][family]
        native_identities[phase] = dict(item, build_receipt=item["receipt"],
                                        build_receipt_sha256=item["receipt_sha256"])
    controls = {"mapping": 0, "num_splits": 0}
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    event_end.record()
    event_end.synchronize()
    rows = []
    with torch.inference_mode():
        for phase in policy.PHASES:
            case_order = policy.CASES if plan["round_id"] == 0 else tuple(reversed(policy.CASES))
            for case in case_order:
                observe("case_start", case=case, phase=phase)
                capture = captures[(case, phase)]
                tensors = restore_inputs(args.capture_dir, capture)
                native = prepare_call(load(native_identities[phase]), tensors, controls, capture, keys)
                for trial, config in enumerate(trials):
                    if config["phase"] != phase:
                        continue
                    id_ = config["candidate_id"]
                    item = builds[id_]
                    policy.require(policy.sha(Path(item["b_plan"])) == item["b_plan_sha256"], "B proof changed since compile")
                    row = dict(item, case=case, phase=phase, controls=config["controls"],
                               capture_sha256=capture["full_tensor_file_sha256"], capture_step=capture["step"],
                               native_module_sha256=native_identities[phase]["module_sha256"],
                               native_controls=controls, native_resources=native["resources"], pairs=[])
                    if item["state"] == "complete":
                        candidate = prepare_call(load(item), tensors, config["controls"], capture, keys)
                        roles = {"official_default": native, "candidate": candidate}
                        for side in roles.values():
                            side["output"].fill_(float("nan"))
                            side["lse"].fill_(float("nan"))
                            side["invoke"]()
                        torch.cuda.synchronize()
                        checks = check_all(candidate, native)
                        row.update(state="passed" if checks["ok"] else "numerical_failed", checks=checks,
                                   full_output_and_lse_checked=True, resources=candidate["resources"],
                                   workspace_spec=candidate["workspace_spec"])
                        if not checks["ok"]:
                            output = args.out / f"failed_{id_}_{case}.pt"
                            torch.save({f"{role}_{name}": side[name].cpu() for role, side in roles.items()
                                        for name in ("output", "lse")}, output)
                            row.update(failed_O_LSE_file=output.name, failed_O_LSE_sha256=policy.sha(output))
                        else:
                            observe("timing_start", case=case, phase=phase, candidate_id=id_)
                            for role in policy.role_order(plan["round_id"], trial, case, 0):
                                roles[role]["invoke"]()
                            torch.cuda.synchronize()
                            for pair in range(policy.N_PAIRS):
                                order = policy.role_order(plan["round_id"], trial, case, pair)
                                sample = dict(pair=pair, role_order=order)
                                for role in order:
                                    event_start.record()
                                    roles[role]["invoke"]()
                                    event_end.record()
                                    event_end.synchronize()
                                    metric = "native_ns" if role == "official_default" else "candidate_ns"
                                    sample[metric] = event_start.elapsed_time(event_end) * 1e6
                                row["pairs"].append(sample)
                        del candidate, roles, side
                    else:
                        policy.require(item["state"] == "compile_failed", "Unrecognized build outcome")
                    append(args.out / "rows.jsonl", row)
                    rows.append(row)
                    print(json.dumps(dict(case=case, phase=phase, candidate_id=id_, state=row["state"])), flush=True)
                del native, tensors
    observe("end")
    evaluation = dict(schema=policy.EVAL_SCHEMA, state="complete", plan_sha256=policy.sha(args.plan),
                      round_id=plan["round_id"], baseline=policy.BASELINE, use="development_only",
                      measurement="cuda_event_single_attention_call_including_combine",
                      regime="same retained input within cell, actual strides restored, warmup before each timed cell",
                      capture_index_sha256=policy.CAPTURE_INDEX_SHA256, build_batch_sha256=policy.sha(args.builds),
                      raw_rows_sha256=policy.sha(args.out / "rows.jsonl"), rows=rows)
    policy.validate_evaluation(evaluation, plan, args.plan)
    policy.write_new(args.out / "evaluation.json", evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("build", "measure"):
        command = subparsers.add_parser(action)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    for option in ("emission", "workspace"):
        subparsers.choices["build"].add_argument(f"--{option}", type=Path, required=True)
    subparsers.choices["build"].add_argument("--run-prefix", required=True)
    for option in ("builds", "capture-dir", "base-manifest"):
        subparsers.choices["measure"].add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    policy.require(not args.out.exists(), "Preserve prior build/acquisition output")
    plan = policy.read(args.plan)
    trials = policy.validate_plan(plan)
    try:
        (build if args.action == "build" else measure)(args, plan, trials)
    except BaseException:
        if args.out.is_dir():
            policy.write_new(args.out / "failure.json", {"state": "failed_preserved", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
