"""Solve and lower bounded CUDA/CuTe schedules for Qwen3 causal prefill."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_spark_pin_v6" / "vendor"
VENDOR_SHA256 = "919ac9c0333363d56fede38963da44266ea5d996f9ec178da4c7c55f02141e45"
UPSTREAM = "49774b31c8d1172a5b28f72e9aed0f322bf4ccba"
HEADER = "csrc/flash_attn/src/flash_fwd_kernel.h"
GEOMETRIES = (
    (32, 64, 2), (64, 32, 4), (64, 64, 4), (64, 128, 4),
    (128, 32, 4), (128, 64, 4), (128, 64, 8), (128, 128, 8),
)
SHARED_LIMIT = 96 * 1024
SEED_IDS = (
    "pf_m128n32w4_reload_forward_late",
    "pf_m128n64w8_reload_forward_early",
    "pf_m64n64w4_alias_forward_late",
    "pf_m64n128w4_reload_reverse_early",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def candidate_specs():
    for (m, n, warps), placement, traversal, pipeline in itertools.product(
        GEOMETRIES, ("reload", "alias"), ("forward", "reverse"), ("early", "late", "preload")
    ):
        # Preloading a complete K MMA fragment is an intentionally narrow
        # N32 experiment. Shared and register footprints are independent.
        if pipeline == "preload" and (n != 32 or placement != "reload"):
            continue
        yield {
            "id": f"pf_m{m}n{n}w{warps}_{placement}_{traversal}_{pipeline}",
            "block_m": m,
            "block_n": n,
            "warps": warps,
            "q_placement": placement,
            "traversal": traversal,
            "pipeline": pipeline,
        }


def replay(order, spec):
    """Check logical waits and shared-memory reuse for both uniform branches."""
    staged = spec["pipeline"] == "preload"
    early = spec["pipeline"] == "early"
    for next_tile in (False, True):
        # Loop entry wait_group 0 and CTA barrier drain the previous K copy
        # and retire the previous PV's access to the reusable V tile.
        groups = []
        k_last_shared_read = False
        k_released = False
        v_ready = False
        scores_ready = False
        probability_ready = False
        for action in order:
            if action == "issue_v":
                groups.append("v")
            elif action == "stage_k":
                if not staged or k_released:
                    raise ValueError("K must be read before its shared tile is released")
                k_last_shared_read = True
            elif action == "qk":
                if staged != k_last_shared_read or (not staged and k_released):
                    raise ValueError("QK must use the current K tile or its staged fragment")
                k_last_shared_read = True
                scores_ready = True
            elif action == "release_k":
                if not k_last_shared_read:
                    raise ValueError("All K shared reads must finish before its next write")
                k_released = True
            elif action == "issue_next_k" and next_tile:
                if not k_released:
                    raise ValueError("K overwrite lacks a preceding CTA barrier")
                groups.append("k")
            elif action == "wait_v":
                pending = 1 if next_tile and not early else 0
                completed = groups[:len(groups) - pending]
                groups = groups[len(groups) - pending:]
                if "v" not in completed or "v" in groups:
                    raise ValueError("V readiness must include its committed copy group")
                v_ready = True
                # This wait is followed by a CTA barrier. In the early mode it
                # also proves that every warp has finished its K shared reads.
                k_released = k_released or k_last_shared_read
            elif action == "softmax" and not scores_ready:
                raise ValueError("Softmax reads unavailable scores")
            elif action == "convert_p":
                if order.index("softmax") >= order.index(action):
                    raise ValueError("Probability conversion must follow softmax")
                probability_ready = True
            elif action == "pv" and not (probability_ready and v_ready):
                raise ValueError("PV requires complete probabilities and V")
        if groups != (["k"] if next_tile else []):
            raise ValueError("Only the prefetched next K may remain pending at loop exit")
    return {"next_tile": "V ready, only next K pending", "final_tile": "V ready, no pending writes"}


def solve(spec):
    from ortools.sat.python import cp_model

    m, n, warps = (spec[key] for key in ("block_m", "block_n", "warps"))
    q_bytes = m * 128 * 2
    kv_bytes = 2 * n * 128 * 2
    alias = spec["q_placement"] == "alias"
    shared_bytes = max(q_bytes, kv_bytes) if alias else q_bytes + kv_bytes
    early = spec["pipeline"] == "early"
    staged = spec["pipeline"] == "preload"
    actions = ["issue_v", "qk", "wait_v", "issue_next_k", "softmax", "convert_p", "pv"]
    edges = [
        ("issue_v", "wait_v"), ("issue_v", "qk"), ("qk", "softmax"),
        ("softmax", "convert_p"), ("convert_p", "pv"),
        ("wait_v", "pv"), ("issue_next_k", "pv"),
    ]
    if early:
        edges.extend((("qk", "wait_v"), ("wait_v", "issue_next_k"), ("wait_v", "softmax")))
    else:
        actions.append("release_k")
        edges.extend((
            ("issue_v", "release_k"), ("release_k", "issue_next_k"),
            ("issue_next_k", "wait_v"), ("convert_p", "wait_v"),
        ))
        if staged:
            actions.append("stage_k")
            edges.extend((
                ("issue_v", "stage_k"), ("stage_k", "release_k"), ("issue_next_k", "qk"),
            ))
        else:
            edges.extend((("qk", "release_k"), ("issue_next_k", "softmax")))

    model = cp_model.CpModel()
    pos = {action: model.NewIntVar(0, len(actions) - 1, action) for action in actions}
    model.AddAllDifferent(list(pos.values()))
    for before, after in edges:
        model.Add(pos[before] < pos[after])
    shared = model.NewIntVar(0, SHARED_LIMIT, "dynamic_shared_bytes")
    model.Add(shared == shared_bytes)
    # Minimize K prefetch position within each legal design point. This is a
    # source order priority, not a predicted device time.
    model.Minimize(pos["issue_next_k"])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"Could not solve {spec['id']}: {solver.StatusName(status)}")
    positions = {action: solver.Value(variable) for action, variable in pos.items()}
    order = sorted(actions, key=positions.get)
    proof = replay(order, spec)
    source_features = {
        "block_m": m,
        "block_n": n,
        "warps": warps,
        "dynamic_shared_bytes": solver.Value(shared),
        "q_tile_bytes": q_bytes,
        "kv_tile_bytes": kv_bytes,
        "q_in_regs": int(alias),
        "q_k_alias": int(alias),
        "reverse_tiles": int(spec["traversal"] == "reverse"),
        "staged_k": int(staged),
        "defer_v_wait": int(not early),
        "k_prefetch_to_pv_steps": positions["pv"] - positions["issue_next_k"],
        "p_live_steps": positions["pv"] - positions["convert_p"],
        "additional_cta_barriers": int(not early),
    }
    return {
        **spec,
        "phase": "prefill",
        "mapping": 1,
        "num_splits": 0,
        "source_features": source_features,
        "schedule": {
            "solver": "OR-Tools CP-SAT",
            "status": solver.StatusName(status),
            "precedence": edges,
            "positions": positions,
            "order": order,
            "objective": "earliest legal next-K commit",
            "objective_value": solver.Value(pos["issue_next_k"]),
            "shared_limit_bytes": SHARED_LIMIT,
            "shared_layout": {
                "q": [0, q_bytes],
                "k": [0 if alias else q_bytes, n * 128 * 2],
                "v": [(0 if alias else q_bytes) + n * 128 * 2, n * 128 * 2],
            },
            "shared_layout_units": "byte offset and size",
            "loop_entry": "wait_group 0, then CTA barrier before shared K/V use",
            "q_alias_contract": "FA2 prologue loads Q registers before K can overwrite shared Q",
            "replay": proof,
        },
    }


def enumerate_plans():
    """Return the bounded feasible catalog with source schedule features."""
    return [solve(spec) for spec in candidate_specs()]


def propose_plan(scores: dict[str, float], *, exclude=()):
    """Use measured A predictions as the CP-SAT objective for a fresh B plan."""
    from ortools.sat.python import cp_model

    excluded = set(exclude)
    specs = {spec["id"]: spec for spec in candidate_specs()}
    if set(scores) - specs.keys() or excluded - specs.keys():
        raise ValueError("Feedback contains unknown prefill plan identifiers")
    candidates = []
    for plan_id, score in sorted(scores.items()):
        if plan_id in excluded:
            continue
        if not math.isfinite(score) or abs(score) > 1e8:
            raise ValueError("Finite bounded A scores are required")
        plan = solve(specs[plan_id])
        candidates.append((plan, float(score), round(score * 1_000_000)))
    if not candidates:
        raise ValueError("Feedback must score at least one unseen legal plan")

    model = cp_model.CpModel()
    choice = model.NewIntVar(0, len(candidates) - 1, "schedule_choice")
    cost = model.NewIntVar(min(row[2] for row in candidates), max(row[2] for row in candidates), "A_cost")
    shared = model.NewIntVar(0, SHARED_LIMIT, "dynamic_shared_bytes")
    block_m = model.NewIntVar(32, 128, "query_tile")
    block_n = model.NewIntVar(32, 128, "kv_tile")
    warps = model.NewIntVar(2, 8, "warp_count")
    alias = model.NewBoolVar("q_k_alias")
    reverse = model.NewBoolVar("reverse_tiles")
    staged = model.NewBoolVar("stage_k")
    deferred = model.NewBoolVar("defer_v_wait")
    actions = sorted({action for plan, _, _ in candidates for action in plan["schedule"]["order"]})
    positions = [model.NewIntVar(-1, len(actions) - 1, action) for action in actions]
    feasible_rows = []
    for index, (plan, _, ticks) in enumerate(candidates):
        features = plan["source_features"]
        feasible_rows.append([
            index, ticks, features["dynamic_shared_bytes"], features["block_m"], features["block_n"],
            features["warps"], features["q_k_alias"], features["reverse_tiles"], features["staged_k"],
            features["defer_v_wait"], *(plan["schedule"]["positions"].get(action, -1) for action in actions),
        ])
    model.AddAllowedAssignments(
        [choice, cost, shared, block_m, block_n, warps, alias, reverse, staged, deferred, *positions],
        feasible_rows,
    )
    # A's predicted cost is primary. A stable catalog ID breaks rounded ties.
    model.Minimize(cost * len(candidates) + choice)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"A-driven B proposal failed: {solver.StatusName(status)}")
    index = solver.Value(choice)
    plan, score, ticks = candidates[index]
    return {
        "schema": "qwen3-long-prefill-A-driven-B-proposal-v14",
        "id": plan["id"],
        "status": solver.StatusName(status),
        "plan": plan,
        "score": score,
        "score_ticks": ticks,
        "score_scale": 1_000_000,
        "objective": "Minimize A predicted cost, stable ID breaks rounded ties",
        "excluded_ids": sorted(excluded),
        "scored_ids": [{"id": p["id"], "score": value, "ticks": t} for p, value, t in candidates],
        "resolved_positions": {action: solver.Value(position) for action, position in zip(actions, positions)},
    }


def lower_loop(plan):
    early = plan["pipeline"] == "early"
    staged = plan["pipeline"] == "preload"
    actions = {
        "issue_v": """FLASH_NAMESPACE::copy<true, Is_even_K>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();""",
        "stage_k": """Tensor k_register_copy = smem_thr_copy_K.retile_D(tSrK);
        CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(k_register_copy));
        cute::copy(smem_tiled_copy_K, tSsK, k_register_copy);""",
        "qk": f"""FLASH_NAMESPACE::gemm<Kernel_traits::Is_Q_in_regs, {str(staged).lower()}>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);""",
        "release_k": """if (has_next) {
            __syncthreads();
        }""",
        "issue_next_k": """if (has_next) {
            FLASH_NAMESPACE::copy<true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            cute::cp_async_fence();
        }""",
        "wait_v": "FLASH_NAMESPACE::cp_async_wait<0>();\n        __syncthreads();" if early else
        """if (has_next) {
            // V was committed first. Leave at most the newer K group pending.
            FLASH_NAMESPACE::cp_async_wait<1>();
        } else {
            FLASH_NAMESPACE::cp_async_wait<0>();
        }
        __syncthreads();""",
        "softmax": """softmax.template softmax_rescale_o<false, false>(
            acc_s, acc_o, params.scale_softmax_log2);""",
        "convert_p": """Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));""",
        "pv": """FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
            smem_tiled_copy_V, smem_thr_copy_V);""",
    }
    order = plan["schedule"]["order"]
    body = "\n\n".join(f"        // B schedule: {action}\n        {actions[action]}" for action in order)
    return f"""    // Generated unmasked loop. The pinned masked prefix and epilogue are intact.
    for (; n_block >= n_block_min; --n_block) {{
        const bool has_next = n_block > n_block_min;
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{{}});
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

{body}
    }}

"""


def rowblock_source(header, plan):
    signature = header.index("\ninline __device__ void compute_attn_1rowblock(")
    start = header.rindex("\ntemplate<", 0, signature)
    split = header.index("\ninline __device__ void compute_attn_1rowblock_splitkv(")
    end = header.rindex("\ntemplate<", signature, split)
    rowblock = header[start:end].replace("compute_attn_1rowblock(", "long_prefill_rowblock(", 1)
    loop_start = rowblock.index("    // These are the iterations where we don't need masking on S")
    epilogue = rowblock.index("    // Epilogue", loop_start)
    rowblock = rowblock[:loop_start] + lower_loop(plan) + rowblock[epilogue:]
    signature_end = rowblock.index("{", rowblock.index("long_prefill_rowblock(")) + 1
    guard = """
    static_assert(Is_causal && Is_even_MN && Is_even_K);
    static_assert(!Is_dropout && !Is_local && !Has_alibi && !Is_softcap && !Return_softmax);
"""
    rowblock = rowblock[:signature_end] + guard + rowblock[signature_end:]
    return header[:header.index("#pragma once")] + (
        '#pragma once\n#include "flash_fwd_kernel.h"\n\nnamespace FLASH_NAMESPACE {\n'
        + rowblock + "\n} // namespace FLASH_NAMESPACE\n"
    )


def emit(out: Path, plan_id: str, *, family_code: int, vendor: Path = VENDOR, proposal=None):
    """Solve the requested point again and bind its actual lowered source."""
    if type(family_code) is not int or not 200 <= family_code < 2**31:
        raise ValueError("Use a fresh positive V14 family code, at least 200")
    spec = next((spec for spec in candidate_specs() if spec["id"] == plan_id), None)
    if spec is None:
        raise ValueError(f"Unknown prefill plan {plan_id}")
    plan = solve(spec)
    if proposal is not None and (
        proposal["id"] != plan_id or proposal["status"] != "OPTIMAL" or proposal["plan"] != plan
    ):
        raise ValueError("A-driven B proposal does not bind the emitted prefill plan")
    vendor_manifest = vendor / "manifest.json"
    if sha(vendor_manifest) != VENDOR_SHA256:
        raise ValueError("Exact pinned vendor manifest required")
    pin = json.loads(vendor_manifest.read_bytes())
    header = vendor / HEADER
    if pin["commit"] != UPSTREAM or sha(header) != pin["files"][HEADER]["sha256"]:
        raise ValueError("Pinned FA2 header bytes changed")
    parent = ROOT / "native_model_v11" / "emitted_v3" / "prefill_natural64"
    v11 = json.loads((parent / "manifest.json").read_bytes())
    for name in ("binding.cpp", "common.cuh"):
        if sha(parent / name) != v11["source_sha256"][name]:
            raise ValueError(f"V11 ABI parent changed: {name}")
    m, n, warps = (plan[key] for key in ("block_m", "block_n", "warps"))
    alias = int(plan["q_placement"] == "alias")
    reverse = int(plan["traversal"] == "reverse")
    config = (
        f"#pragma once\n#define MODEL_FAMILY_CODE {family_code}\n"
        f"#define PREFILL_BLOCK_M {m}\n#define PREFILL_BLOCK_N {n}\n#define PREFILL_NWARPS {warps}\n"
        f"#define PREFILL_Q_IN_REGS {alias}\n#define PREFILL_SHARE_QK {alias}\n"
        f"#define PREFILL_REVERSE_TILES {reverse}\n"
    )
    template = ROOT / "native_model_long_prefill_v14" / "prefill.cu"
    rowblock = rowblock_source(header.read_text(encoding="utf-8"), plan)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("binding.cpp", "common.cuh"):
        shutil.copyfile(parent / name, out / name)
    shutil.copyfile(template, out / "prefill.cu")
    (out / "config.cuh").write_bytes(config.encode())
    (out / "long_prefill_rowblock.cuh").write_bytes(rowblock.encode())
    manifest = {
        **plan,
        "schema": "qwen3-long-context-prefill-B-v14",
        "state": "source_only_not_compiled_or_GPU_validated",
        "family": plan_id,
        "family_code": family_code,
        "candidate_mapping": 1,
        "vendor_manifest_sha256": VENDOR_SHA256,
        "upstream_commit": UPSTREAM,
        "pinned_rowblock_parent_sha256": sha(header),
        "v11_parent_manifest_sha256": sha(parent / "manifest.json"),
        "generator_sha256": sha(Path(__file__)),
        "source_sha256": {path.name: sha(path) for path in sorted(out.iterdir())},
        "resource_keys": v11["resource_keys"],
        "abi": v11["abi"],
        "layout": "Actual strided BNHD Hq32/Hkv8/D128 with no packing",
        "scope": "Fully tiled BF16 equal-length dense causal prefill with no past cache",
        "grid_order": "Four adjacent logical CTAs share a KV head. Physical execution order is unspecified.",
        "complete_output": "Contiguous BNHD O and FP32 natural-log LSE",
        "actual_compiled_resources_required": True,
    }
    if proposal is not None:
        manifest["A_driven_B_proposal"] = proposal
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--emit", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--family-code", type=int)
    parser.add_argument("--vendor", type=Path, default=VENDOR)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(enumerate_plans(), indent=2))
    else:
        print(json.dumps(emit(args.emit, args.id, family_code=args.family_code, vendor=args.vendor), indent=2))


if __name__ == "__main__":
    main()
