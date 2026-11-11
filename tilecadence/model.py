"""V5.2 exact4KiB storage and ready-aware last-commit-to-wait overlap.

This is a finite source-dependency/liveness model, not a physical RF allocator
or a simulator of compiled instruction completion. Costs are caller supplied.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

SLAB_BYTES = 4096
OUTPUT_STRIPE_BYTES = 16384
REQUIRED_PROFILES = ("copy_q", "copy_k", "copy_v64", "load_q", "load_k", "load_v64",
                     "qk", "stats_emit_p_first", "stats_emit_p_next", "recover_p",
                     "pv64_first", "rescale_o64", "pv64_next", "normalize_store64", "store_lse")
SOURCE_KINDS = {"rf_kib_steps", "rf_hold_kib_steps", "o_kib_steps", "o_overlap_steps",
                "g2s_to_s2r_gap_steps", "dependency_gap_steps", "rf_peak_kib",
                "release_barrier_count", "g2s_interposed_actions", "consumer_wait_count",
                "last_commit_wait_hmma_sites", "last_commit_wait_ex2_sites"}
VALUE_ROLES = {"q", "k", "v", "s", "p", "m", "l", "alpha"}
ACTION_KINDS = {"copy_g2s", "load_s2r", "qk", "stats_first", "stats_next", "pv_first", "pv_next", "rescale_o", "normalize_store", "store_lse"}
MODEL_REVISION = "v52-fourk-ready-exact-read-mask-last-commit-overlap"
MATH_SITE_WEIGHTS={"qk":(32,0),"pv_first":(16,0),"pv_next":(16,0),
                   "stats_first":(0,16),"stats_next":(0,18)}
CP_CERTIFICATE_KEYS=("id","read_mask_before","read_mask_after","ready_cutoff_before",
    "ready_cutoff_after","wait_before_load","release_before_write",
    "last_commit_hmma_before","last_commit_hmma_after","last_commit_ex2_before","last_commit_ex2_after",
    "last_commit_wait_hmma_sites","last_commit_wait_ex2_sites")
LEGACY_SOURCE_KEYS = {"rf_kib_steps", "rf_hold_kib_steps", "o_kib_steps", "o_overlap_steps", "g2s_to_s2r_gap_steps", "dependency_gap_steps", "rf_peak_kib",
    *[f"rf_hold_{role}_kib_steps" for role in VALUE_ROLES], *[f"g2s_{role}_gap_steps" for role in ("q", "k", "v")]}
SOURCE_ALIASES = {**{f"rf_hold_{role}_kib_steps": {"kind": "rf_hold_kib_steps", "roles": [role]} for role in sorted(VALUE_ROLES)},
    **{f"g2s_{role}_gap_steps": {"kind": "g2s_to_s2r_gap_steps", "roles": [role]} for role in ("q", "k", "v")},
    **{f"release_before_{role}_copy": {"kind": "release_barrier_count", "roles": [role], "write_kinds": ["copy_g2s"]} for role in ("q", "k", "v")},
    "release_before_close": {"kind": "release_barrier_count", "write_kinds": ["normalize_store"]},
    **{f"g2s_{role}_{label}_between": {"kind": "g2s_interposed_actions", "roles": [role], "kinds": kinds}
       for role in ("q", "k", "v") for label, kinds in (("qk", ["qk"]), ("stats", ["stats_first", "stats_next"]),
            ("pv", ["pv_first", "pv_next"]), ("close", ["normalize_store"]))}}


def source_definition(name):
    require(name in SOURCE_KINDS or name in SOURCE_ALIASES, "Unknown structural source feature name")
    return dict(SOURCE_ALIASES[name]) if name in SOURCE_ALIASES else {"kind": name}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(x, minimum=0):
    return isinstance(x, int) and not isinstance(x, bool) and x >= minimum


def _validate_source_definition(source):
    require(isinstance(source, dict) and set(source) <= {"kind", "roles", "tiers", "write_kinds", "kinds"}
            and source.get("kind") in SOURCE_KINDS, "Unknown source-order feature")
    if "roles" in source:
        require(source["kind"] in {"rf_hold_kib_steps", "g2s_to_s2r_gap_steps", "dependency_gap_steps", "release_barrier_count", "g2s_interposed_actions"}
                and isinstance(source["roles"], list) and bool(source["roles"])
                and set(source["roles"]) <= VALUE_ROLES and len(source["roles"]) == len(set(source["roles"])), "Invalid source role filter")
    if "tiers" in source:
        require(source["kind"] == "dependency_gap_steps" and isinstance(source["tiers"], list)
                and bool(source["tiers"]) and set(source["tiers"]) <= {"reg", "smem"}
                and len(source["tiers"]) == len(set(source["tiers"])), "Invalid source tier filter")
    for field, kind, allowed in (("write_kinds", "release_barrier_count", {"copy_g2s", "normalize_store"}),
                                 ("kinds", "g2s_interposed_actions", ACTION_KINDS)):
        if field in source:
            require(source["kind"] == kind and isinstance(source[field], list) and bool(source[field])
                    and set(source[field]) <= allowed and len(source[field]) == len(set(source[field])), "Invalid lowering/interposed action filter")
    return source


def _validate_source_bounds(bounds, spec):
    require(isinstance(bounds, dict), "Source bounds must be a mapping")
    priced = {term["feature"]: term["source"] for term in spec["terms"] if "source" in term} if spec else {}
    require(all(name in priced and priced[name] == source_definition(name) for name in bounds),
            "Each named bound must have its exact structural source objective term, including zero coefficients")
    for kind, interval in bounds.items():
        require(isinstance(interval, (list, tuple)) and len(interval) == 2 and
                integer(interval[0]) and integer(interval[1]) and interval[0] <= interval[1],
                "Source bounds must be closed ordered nonnegative integer intervals")
    return {kind: list(interval) for kind, interval in bounds.items()}


def _validate_objective_spec(spec):
    if spec is None:
        return None
    spec = json.loads(json.dumps(spec))
    require(set(spec) == {"unit", "constant", "terms", "provenance"}, "Affine objective needs unit/constant/terms/provenance")
    require(spec["unit"] in ("cost_ticks", "effective_ns", "cycles") and
            isinstance(spec["constant"], int) and not isinstance(spec["constant"], bool), "Invalid affine unit/base")
    require(isinstance(spec["terms"], list) and isinstance(spec["provenance"], dict) and bool(spec["provenance"]), "Explicit affine terms/provenance required")
    names = set()
    for term in spec["terms"]:
        require(set(term) in ({"feature", "coefficient", "match"}, {"feature", "coefficient", "source"}) and isinstance(term["feature"], str)
                and term["feature"] and term["feature"] not in names, "Unique affine features required")
        names.add(term["feature"])
        require(isinstance(term["coefficient"], int) and not isinstance(term["coefficient"], bool), "Affine coefficients must be integer, including zero")
        if "source" in term:
            _validate_source_definition(term["source"])
            continue
        match = term["match"]
        require(isinstance(match, dict) and set(match) <= {"name", "kind", "role", "instance_min", "instance_max", "tile", "stripe"}, "Features may match action attributes, never kernel IDs")
        for field, value in match.items():
            if field in ("instance_min", "instance_max"):
                require(integer(value), "Invalid instance filter")
            elif field in ("tile", "stripe"):
                require(value is None or integer(value) and value in (0, 1), "Invalid tile/stripe filter")
            else:
                candidates = value if isinstance(value, list) else [value]
                require(bool(candidates) and all(isinstance(x, str) and x for x in candidates)
                        and len(set(candidates)) == len(candidates), "Action filters require nonempty strings or distinct string lists")
    return spec


def action_matches(action, match, values):
    for field, value in match.items():
        if field == "instance_min":
            if action["instance"] < value:
                return False
        elif field == "instance_max":
            if action["instance"] > value:
                return False
        elif field == "role":
            role = values[action["value"]].role if action["value"] is not None else None
            if role not in (value if isinstance(value, list) else [value]):
                return False
        elif action[field] not in (value if isinstance(value, list) else [value]):
            return False
    return True


def objective_features(p, selected, result=None):
    actions = {a["id"]: a for a in p.actions}
    items = [actions[e if isinstance(e, str) else e["id"]] for e in selected]
    values = {v.name: v for v in p.values}
    if p.objective_spec is None:
        return {}
    require(result is not None or not any("source" in t for t in p.objective_spec["terms"]), "Source features require full allocation/binding result")
    return {term["feature"]: (_source_value(p, result, term["source"]) if "source" in term else
                             sum(action_matches(a, term["match"], values) for a in items))
            for term in p.objective_spec["terms"]}


def evaluate_objective(p, selected, result=None):
    actions = {a["id"]: a for a in p.actions}
    items = [actions[e if isinstance(e, str) else e["id"]] for e in selected]
    if p.objective_spec is None:
        return sum(a["profile_cost"] for a in items)
    counts = objective_features(p, selected, result)
    return p.objective_spec["constant"] + sum(term["coefficient"]*counts[term["feature"]] for term in p.objective_spec["terms"])


@dataclass(frozen=True)
class Cost:
    # Historical field name retained for callers: this is an objective
    # coefficient, NOT a source interval length or claimed hardware latency.
    duration: int
    kind: str
    provenance: Mapping[str, Any]
    unit: str = "cost_ticks"

    def __post_init__(self):
        require(integer(self.duration) and self.unit in ("cost_ticks", "cycles", "effective_ns"), "Supply nonnegative integer coefficients with explicit units")
        require(self.kind in ("measured", "derived", "synthetic") and bool(self.provenance), "Explicit cost kind/provenance required")


@dataclass(frozen=True)
class Value:
    name: str
    role: str
    tile: int | None
    stripe: int | None
    logical_bytes: int
    register_bytes: int
    layout: str


@dataclass(frozen=True)
class Problem:
    values: tuple[Value, ...]
    actions: tuple[Mapping[str, Any], ...]
    profiles: Mapping[str, Cost]
    arena_slabs: int
    register_budget_bytes: int
    precedence_edges: tuple[tuple[str, str], ...]
    state_chains: tuple[tuple[str, ...], ...]
    kv_order: tuple[int, ...]
    workspace_reg_bytes: Mapping[str, int]
    required_action_ids: tuple[str, ...]
    shared_alignment_slabs_by_role: Mapping[str, int]
    objective_spec: Mapping[str, Any] | None
    provenance: Mapping[str, Any]


def build_problem(profiles: Mapping[str, Cost] | None = None, *, arena_slabs=4, register_budget_bytes=98304,
                  allow_score_spill=True, allow_probability_spill=True, max_qk_instances=2,
                  max_p_recoveries=1, workspace_reg_bytes=None, profile_overrides=None,
                  kv_order=(0, 1), extra_precedence=(), required_action_ids=(),
                  execution_mode="source_order", provenance=None, backend_spec=None, objective_spec=None,
                  close_workspace_smem_bytes=8192, source_feature_bounds=None, forbidden_source_orders=(),
                  full_value_parent=False):
    require(arena_slabs in (4, 8) and integer(register_budget_bytes, 1), "Bounded4KiB-unit arena and positive RF budget required")
    require(max_qk_instances in (1, 2) and integer(max_qk_instances, 1), "QK bound must be one or two")
    require(max_p_recoveries in (0, 1) and integer(max_p_recoveries), "P recovery bound must be zero or one")
    require(tuple(kv_order) in ((0, 1), (1, 0)), "Exactly two ordered KV tiles required")
    require(execution_mode in ("source_order", "completion_enforced"), "Unknown backend execution mode")
    require(integer(close_workspace_smem_bytes) and close_workspace_smem_bytes in (0, 8192), "Close workspace must be zero or one declared 8KiB slab")
    backend = dict(backend_spec or {})
    require(isinstance(full_value_parent,bool),"Declare split64 or shared-parent128 V transfer explicitly")
    require(backend.get("copy_semantics", "async_issue_ready_mask_wait") == "async_issue_ready_mask_wait", "V5 requires exact ready-cutoff waits and pending shared-read masks")
    align = {role: 1 for role in ("q", "k", "v", "s", "p", "m", "l", "alpha")}
    align.update(backend.get("shared_alignment_slabs_by_role", {}))
    require(all(integer(x, 1) for x in align.values()), "Backend alignment must use positive slab multiples")
    require(set(align) == {"q", "k", "v", "s", "p", "m", "l", "alpha"}, "Unknown backend alignment role")
    replication = {"q": 1, "k": 4, "v": 4, "s": 1, "p": 1, "m": 4, "l": 4, "alpha": 4}
    replication.update(backend.get("register_replication_by_role", {}))
    require(set(replication) == set(align) and all(integer(x, 1) for x in replication.values()), "Invalid declared backend RF replication")
    require(backend.get("row_sum_representation", "full_sum_quad_replicated") == "full_sum_quad_replicated",
            "V3 equations require full row sums; a lane-partial backend needs a different graph")
    needed = set(REQUIRED_PROFILES) - ({"recover_p"} if max_p_recoveries == 0 else set())
    if full_value_parent:
        needed.remove("copy_v64");needed.add("copy_v128")
    if allow_score_spill:
        needed.update(("store_s", "load_s"))
    if allow_probability_spill:
        needed.update(("store_p", "load_p"))
    spec = _validate_objective_spec(objective_spec)
    bounds = _validate_source_bounds(dict(source_feature_bounds or {}), spec)
    if profiles is None:
        require(spec is not None, "Supply explicit objective coefficients or an affine objective")
        profiles = {key: Cost(0, "derived", {"scope": "Unpriced primitive vocabulary; ignored by affine whole-schedule proxy", "not_a_primitive_latency": True}, spec["unit"])
                    for key in needed}
    profiles = dict(profiles)
    require(all(isinstance(c, Cost) for c in profiles.values()), "Supply Cost objects")
    units = {c.unit for c in profiles.values()}
    require(len(units) == 1 and (spec is None or units == {spec["unit"]}), "Mixed objective units are not allowed")
    require(needed <= profiles.keys(), "Missing explicit action costs: " + str(sorted(needed - profiles.keys())))
    workspace = dict(workspace_reg_bytes or {})
    require(all(integer(x) for x in workspace.values()), "Workspace demands must be nonnegative integer bytes")
    overrides = dict(profile_overrides or {})
    require(set(overrides.values()) <= profiles.keys(), "Override names an unavailable cost")
    def value(name, role, tile, stripe, logical, layout):
        return Value(name, role, tile, stripe, logical, logical*replication[role], layout)
    values = [value("Q", "q", None, None, 16384, "M64xD128 full A operand")]
    for j in (0, 1):
        values.extend((value(f"K{j}", "k", j, None, 8192, "D128xN32 B operand; declared backend replication"),
                       value(f"S{j}", "s", j, None, 8192, "M64xN32 FP32 accumulator"),
                       value(f"P{j}", "p", j, None, 4096, "M64xN32 input-dtype A operand"),
                       value(f"m{j}", "m", j, None, 256, "FP32 post-update row maximum; declared backend replication"),
                       value(f"l{j}", "l", j, None, 256, "FP32 full row sum; declared backend replication")))
        if full_value_parent:
            values.append(value(f"V{j}","v",j,None,8192,"N32xD128 shared parent; two exact contiguous4KiB views"))
        for s in (0, 1):
            values.append(value(f"V{j}_{s}", "v", j, s, 4096, "N32xDv64 B operand; declared backend replication"))
    later = kv_order[1]
    values.append(value("alpha1", "alpha", later, None, 256, "FP32 online rescale factor; declared backend replication"))
    actions = []

    def add(name, kind, inputs, outputs, base, maximum=1, mandatory=False, tile=None, stripe=None, first=False):
        for instance in range(maximum):
            fallback = "qk_repeat" if kind == "qk" and instance and "qk_repeat" in profiles else base
            key = overrides.get(f"{name}#{instance}", overrides.get(name, fallback))
            require(key in profiles, f"Missing cost {key}")
            actions.append({"id": f"{name}#{instance}", "name": name, "instance": instance,
                "kind": kind, "input_refs": list(inputs), "output_refs": list(outputs),
                "profile_key": key, "duration": 1, "profile_cost": profiles[key].duration, "mandatory": mandatory,
                "value": outputs[0][0] if len(outputs) == 1 else None, "tile": tile, "stripe": stripe,
                "first": first, "workspace_reg_bytes": workspace.get(name, workspace.get(kind, 0)),
                "workspace_smem_bytes": close_workspace_smem_bytes if kind == "normalize_store" else 0})

    for v in values:
        if v.role in ("q", "k", "v"):
            maximum = 4 if v.role == "q" else 2
            suffix = "v64" if v.role == "v" else v.role
            parent = full_value_parent and v.role=="v" and v.stripe is None
            stripe_view = full_value_parent and v.role=="v" and v.stripe is not None
            if not stripe_view:
                add("copy_"+v.name,"copy_g2s",((v.name,"gmem"),),((v.name,"smem"),),
                    "copy_v128" if parent else "copy_"+suffix,maximum,tile=v.tile,stripe=v.stripe)
            if not parent:
                source_name=f"V{v.tile}" if stripe_view else v.name
                add("load_"+v.name,"load_s2r",((source_name,"smem"),),((v.name,"reg"),),
                    "load_"+suffix,maximum,tile=v.tile,stripe=v.stripe)
        elif (v.role == "s" and allow_score_spill) or (v.role == "p" and allow_probability_spill):
            add("store_" + v.name, "store_r2s", ((v.name, "reg"),), ((v.name, "smem"),), "store_" + v.role,
                1, tile=v.tile)
            add("load_" + v.name, "load_s2r", ((v.name, "smem"),), ((v.name, "reg"),), "load_" + v.role,
                2, tile=v.tile)
    first, second = kv_order
    for step, j in enumerate(kv_order):
        add(f"qk_{j}", "qk", (("Q", "reg"), (f"K{j}", "reg")), ((f"S{j}", "reg"),),
            "qk", max_qk_instances, tile=j)
        ins = ((f"S{j}", "reg"),) if step == 0 else ((f"S{j}", "reg"), (f"m{first}", "reg"), (f"l{first}", "reg"))
        outs = ((f"m{j}", "reg"), (f"l{j}", "reg")) + (() if step == 0 else (("alpha1", "reg"),)) + ((f"P{j}", "reg"),)
        add(f"stats_{j}", "stats_first" if step == 0 else "stats_next", ins, outs,
            "stats_emit_p_first" if step == 0 else "stats_emit_p_next", mandatory=True, tile=j, first=step == 0)
        if max_p_recoveries:
            add(f"recover_p_{j}", "recover_p", ((f"S{j}", "reg"), (f"m{j}", "reg")),
                ((f"P{j}", "reg"),), "recover_p", 1, tile=j)
    chains = []
    for s in (0, 1):
        add(f"pv_{first}_{s}", "pv_first", ((f"P{first}", "reg"), (f"V{first}_{s}", "reg")), (),
            "pv64_first", mandatory=True, tile=first, stripe=s, first=True)
        add(f"rescale_1_{s}", "rescale_o", (("alpha1", "reg"),), (), "rescale_o64", mandatory=True, tile=second, stripe=s)
        add(f"pv_{second}_{s}", "pv_next", ((f"P{second}", "reg"), (f"V{second}_{s}", "reg")), (),
            "pv64_next", mandatory=True, tile=second, stripe=s)
        add(f"close_{s}", "normalize_store", ((f"l{second}", "reg"),), (), "normalize_store64", mandatory=True, stripe=s)
        chains.append((f"pv_{first}_{s}#0", f"rescale_1_{s}#0", f"pv_{second}_{s}#0", f"close_{s}#0"))
    add("store_lse", "store_lse", ((f"m{second}", "reg"), (f"l{second}", "reg")), (), "store_lse", mandatory=True)
    edges = [(f"stats_{first}#0", f"stats_{second}#0")]
    for j in kv_order:
        if max_qk_instances > 1:
            # Canonical recovery: the first QK supplies prefix statistics;
            # another QK is useful only after those statistics have executed.
            edges.append((f"stats_{j}#0", f"qk_{j}#1"))
    for chain in chains:
        edges.extend(zip(chain, chain[1:]))
    def identity(x):
        return x if "#" in x else x + "#0"
    edges.extend((identity(a), identity(b)) for a, b in extra_precedence)
    required = tuple(identity(a) for a in required_action_ids)
    known = {a["id"] for a in actions}
    require(all(a in known and b in known and a != b for a, b in edges), "Unknown/self precedence edge")
    require(set(required) <= known, "Unknown required action")
    require(not (set(workspace) - ({a["name"] for a in actions} | {a["kind"] for a in actions})), "Unknown workspace action")
    return Problem(tuple(values), tuple(actions), dict(profiles), arena_slabs, register_budget_bytes,
        tuple(edges), tuple(chains), tuple(kv_order), workspace, required, align, spec, {
            **dict(provenance or {}), "shape": {"M": 64, "N": 32, "D": 128, "Dv_stripe": 64, "warps": 4, "kv_tiles": 2},
            "model_revision": MODEL_REVISION, "ordering_unit": "source_steps",
            "source_feature_bounds": bounds,
            "forbidden_source_orders": [list(order) for order in forbidden_source_orders],
            "objective_unit": next(iter(units)),
            "backend_spec": {**backend, "shared_slab_bytes": SLAB_BYTES,
                             "shared_alignment_slabs_by_role": align,
                             "register_replication_by_role": replication,
                             "row_sum_representation": "full_sum_quad_replicated",
                             "copy_semantics": "async_issue_ready_mask_wait",
                             "full_value_parent":full_value_parent,
                             "close_workspace_smem_bytes": close_workspace_smem_bytes,
                             "shared_release_semantics": "Emitter inserts address-hazard CTA release before overlapping shared writes; source end is not hardware completion",
                             "scope": "Backend-supplied conservative packing/payload declarations; compiler validation required"},
            "execution_mode": execution_mode, "binary_completion_claim": False,
            "execution_contract": "Source dependency/address intervals; compiled completion, guard coverage and RF release need independent validation",
            "RF_scope": "Operand payload plus variable O stripes and immutable row snapshots; compiler allocation/spills remain authoritative",
            "O_state_bytes_per_stripe": OUTPUT_STRIPE_BYTES, "control_storage_bytes": 0,
            "stats_exactly_once": True, "recovery_has_no_statistics_side_effects": True,
            "objective_scope": "Caller supplied action-count and exact source-order/lifetime affine proxy; actual compiled resources require separate postcompile prediction and validation",
            "limitation": "Finite caller-priced source-order optimum only; not physical pressure relief, measured latency or unrestricted hardware optimality"})


def _occupied(value, tier):
    return value.register_bytes if tier == "reg" else math.ceil(value.logical_bytes / SLAB_BYTES) * SLAB_BYTES


def build_order_search_problem(*, objective_spec, source_feature_bounds,
                               arena_slabs=4, register_budget_bytes=98304,
                               extra_precedence=(), provenance=None, forbidden_source_orders=(),full_value_parent=False):
    """Bounded source design: split-V27 or parent-V25 actions and addresses.

    This helper removes rematerialization and checkpoint choices. Callers must
    calibrate the new ready-aware backend independently; V42 costs are invalid.
    The general graph is retained for future studies, outside this search.
    """
    require(arena_slabs in (4,8), "Exact V52 frontier supports four or eight4KiB units")
    p = build_problem(objective_spec=objective_spec, source_feature_bounds=source_feature_bounds,
        arena_slabs=arena_slabs, register_budget_bytes=register_budget_bytes,
        max_qk_instances=1, max_p_recoveries=0, allow_score_spill=False,
        allow_probability_spill=False, close_workspace_smem_bytes=8192,
        extra_precedence=extra_precedence,
        forbidden_source_orders=forbidden_source_orders,
        full_value_parent=full_value_parent,
        backend_spec={"name": "cuda-cute-v52-fourk-ready-mask-primitives", "copy_semantics": "async_issue_ready_mask_wait"},
        provenance={**dict(provenance or {}), "search_scope": "Mandatory split27 or parent25 shared128 order/address actions; no recovery/checkpoint or hardware-cost calibration implied"})
    actions = tuple(a for a in p.actions if a["instance"] == 0)
    known = {a["id"] for a in actions}
    require(len(actions)==len(known)==(25 if full_value_parent else 27),"Wrong declared V transfer action vocabulary")
    return replace(p, actions=actions, required_action_ids=tuple(sorted(known)),
                   precedence_edges=tuple(edge for edge in p.precedence_edges if set(edge) <= known))


def _workspace_value(action):
    return Value("__workspace:" + action["id"], "workspace", None, action["stripe"],
                 action["workspace_smem_bytes"], 0, "Explicit coalesced-close shared scratch, internal to action")


def _expected_steps(p):
    return 25 if p.provenance["backend_spec"]["full_value_parent"] else 27


def shared_read_span(source,destination):
    """Physical read extent is separate from whole-parent allocation life."""
    read=dict(source)
    if source['role']=='v' and source['stripe'] is None:
        require(destination['role']=='v' and destination['stripe'] in (0,1),'Invalid parent stripe view')
        read.update(id=source['id']+':view'+str(destination['stripe']),
                    offset_bytes=source['offset_bytes']+destination['stripe']*4096,occupied_bytes=4096)
    return read


def lowering_trace(p, r):
    """CPU oracle: wait0-completed copy cutoff and union of distributed reads.

    A wait0 plus CTA acquire completes all earlier committed copy groups.
    Other CTA barriers clear distributed reads but never advance copy readiness.
    The cutoff is an ordinal semantic certificate, not elapsed GPU cycles.
    """
    exchange = "residency_exchange_v78" in p.provenance
    require(p.provenance["execution_mode"] == "source_order"
            and p.provenance["backend_spec"]["close_workspace_smem_bytes"] == 8192,
            "Ready-aware lowering requires source order and shared128 output scratch")
    if exchange:
        require(all(e["kind"] in ACTION_KINDS | {"store_r2s"} for e in r["events"]),
                "Residency exchange supports checkpoint transfers and single-execution math")
    else:
        require(len(p.actions)==_expected_steps(p) and all(a["instance"] == 0 and a["kind"] in ACTION_KINDS for a in p.actions)
                and set(p.required_action_ids) | {a["id"] for a in p.actions if a["mandatory"]} == {a["id"] for a in p.actions},
                "Exact frontier requires split27 or parent25 mandatory V5 shared128 actions")
    copies = {c["id"]: c for c in r["copies"]}
    events = {e["id"]: e for e in r["events"]}
    pending, trace = {}, []
    ready_cutoff, issued_groups, ready_groups = 0, 0, 0
    hmma_gap,ex2_gap=0,0
    def span(c):
        return c["offset_bytes"], c["offset_bytes"]+c["occupied_bytes"]
    def mask(c):
        require(c["offset_bytes"]%SLAB_BYTES==0 and c["occupied_bytes"]%SLAB_BYTES==0,"Invalid physical slab span")
        return ((1 << (c["occupied_bytes"]//SLAB_BYTES))-1) << (c["offset_bytes"]//SLAB_BYTES)
    def union():
        result=0
        for c in pending.values():result |= mask(c)
        return result
    for e in sorted(r["events"], key=lambda e: e["start"]):
        before, before_mask = sorted(pending), union()
        cutoff_before, issued_before, ready_before = ready_cutoff, issued_groups, ready_groups
        hmma_before,ex2_before=hmma_gap,ex2_gap
        writes = [copies[x] for x in e["outputs"] if copies[x]["tier"] == "smem"] if e["kind"] in ("copy_g2s", "store_r2s") else []
        if e["kind"] == "normalize_store":
            require(len(e["workspace_copies"]) == 1, "Shared128 close needs exactly one real scratch span")
            writes = [copies[e["workspace_copies"][0]]]
        require(len(writes) <= 1, "V5 lowering has at most one shared write per action")
        hazards=[[write["id"],identity] for write in writes for identity,c in pending.items()
                 if mask(write)&mask(c)]
        if hazards:pending.clear()  # CTA release completes every earlier distributed read.
        wait=False;producer_end=0
        if e["kind"]=="copy_g2s":issued_groups+=1
        if e["kind"] == "load_s2r":
            require(len(e["inputs"]) == 1, "Shared load requires one source")
            source=copies[e["inputs"][0]]
            producer=events[source["producer"]]
            require(producer["end"] <= e["start"], "Shared load must follow its producer")
            if source["role"] in ("q", "k", "v"):
                require(producer["kind"] == "copy_g2s", "Operand requires a committed input copy")
                producer_end=producer["end"]
                wait=producer_end>ready_cutoff
                if wait:
                    pending.clear()
                    ready_cutoff=e["end"]
                    ready_groups=issued_groups
            else:
                require(exchange and source["role"] in ("s", "p") and producer["kind"] == "store_r2s",
                        "Checkpoint load requires a synchronized checkpoint store")
            read=shared_read_span(source,copies[e['outputs'][0]])
            pending[read["id"]]=read
        elif e["kind"] == "store_r2s":
            pending.clear()  # The checkpoint's post-store CTA barrier does not complete async groups.
        elif e["kind"] == "normalize_store":
            pending.clear()  # Internal shared exchange CTA barrier.
            pending[writes[0]["id"]]=writes[0]
        require(0 <= ready_groups <= issued_groups and issued_groups - ready_groups <= 8
                and (exchange or issued_groups <= 7), "Unsupported async copy-group frontier")
        wait_hmma,wait_ex2=(hmma_gap,ex2_gap) if wait else (0,0)
        if e['kind']=='copy_g2s' or wait:hmma_gap=ex2_gap=0
        else:
            hmma,ex2=MATH_SITE_WEIGHTS.get(e['kind'],(0,0))
            hmma_gap+=hmma;ex2_gap+=ex2
        trace.append({"id": e["id"], "kind": e["kind"],
            "write_role": writes[0]["role"] if writes else None,
            "frontier_before": before, "frontier_after": sorted(pending),
            "read_mask_before":before_mask,"read_mask_after":union(),
            "ready_cutoff_before":cutoff_before,"ready_cutoff_after":ready_cutoff,
            "producer_end":producer_end,"wait_before_load":wait,"acquire_before_load":wait,
            "issued_groups_before":issued_before,"issued_groups_after":issued_groups,
            "ready_groups_before":ready_before,"ready_groups_after":ready_groups,
            "last_commit_hmma_before":hmma_before,"last_commit_hmma_after":hmma_gap,
            "last_commit_ex2_before":ex2_before,"last_commit_ex2_after":ex2_gap,
            "last_commit_wait_hmma_sites":wait_hmma,"last_commit_wait_ex2_sites":wait_ex2,
            "release_before_write": bool(hazards), "shared_read_write_hazards": hazards})
    return trace


def _source_value(p, r, source):
    """Pure CPU extraction; intervals are ordinal/address lives, not GPU time."""
    _validate_source_definition(source)
    kind, roles = source["kind"], source.get("roles", VALUE_ROLES)
    events = {e["id"]: e for e in r["events"]}
    copies = {c["id"]: c for c in r["copies"]}
    regs = [c for c in copies.values() if c["tier"] == "reg"]
    def kib(size):
        require(size % 1024 == 0, "KiB-step source features require exact KiB payloads")
        return size // 1024
    if kind == "release_barrier_count":
        return sum(row["release_before_write"] and
                   row["kind"] in source.get("write_kinds", ("copy_g2s", "normalize_store")) and
                   ("roles" not in source or row["write_role"] in roles) for row in lowering_trace(p, r))
    if kind == "consumer_wait_count":
        return sum(row["wait_before_load"] for row in lowering_trace(p,r))
    if kind in ('last_commit_wait_hmma_sites','last_commit_wait_ex2_sites'):
        return sum(row[kind] for row in lowering_trace(p,r))
    if kind == "rf_kib_steps":
        return sum(kib(c["occupied_bytes"])*(c["end"]-c["start"]) for c in regs) + sum(
            kib(s["bytes"])*(s["end"]-s["start"]) for s in r["output_states"]) + sum(
            kib(e["workspace_reg_bytes"])*(e["end"]-e["start"]) for e in events.values())
    if kind == "rf_hold_kib_steps":
        return sum(kib(c["occupied_bytes"])*(c["end"]-c["available"]) for c in regs if c["role"] in roles)
    if kind == "o_kib_steps":
        return sum(kib(s["bytes"])*(s["end"]-s["start"]) for s in r["output_states"])
    if kind == "o_overlap_steps":
        a, b = r["output_states"]
        return max(0, min(a["end"], b["end"])-max(a["start"], b["start"]))
    if kind in ("g2s_to_s2r_gap_steps", "dependency_gap_steps", "g2s_interposed_actions"):
        result = 0
        for e in events.values():
            for identity in e["inputs"]:
                c = copies[identity]
                producer = events.get(c["producer"])
                if producer is None or c["role"] not in roles:
                    continue
                if kind in ("g2s_to_s2r_gap_steps", "g2s_interposed_actions"):
                    selected = producer["kind"] == "copy_g2s" and e["kind"] == "load_s2r"
                else:
                    selected = c["tier"] in source.get("tiers", ("reg", "smem"))
                if selected:
                    if kind == "g2s_interposed_actions":
                        result += sum(middle["kind"] in source.get("kinds", ACTION_KINDS) and
                                      c["available"] <= middle["start"] and middle["end"] <= e["start"]
                                      for middle in events.values())
                    else:
                        result += e["start"]-c["available"]
        return result
    if kind == "rf_peak_kib":
        return max(sum(kib(c["occupied_bytes"]) for c in regs if c["start"] <= step < c["end"]) +
                   sum(kib(s["bytes"]) for s in r["output_states"] if s["start"] <= step < s["end"]) +
                   sum(kib(e["workspace_reg_bytes"]) for e in events.values() if e["start"] <= step < e["end"])
                   for step in range(r["finish_step"]))
    raise ValueError("Unknown source feature")


def source_feature_values(p, r):
    """Frozen basic vocabulary for compile-aware models; no kernel identifiers.

    Call replay separately on untrusted records. This function does no fitting
    and does not read elapsed timings or compiler resources.
    """
    features = {kind: _source_value(p, r, {"kind": kind}) for kind in sorted(SOURCE_KINDS)}
    features.update({name: _source_value(p, r, definition) for name, definition in SOURCE_ALIASES.items()})
    return features


def problem_from_dict(d):
    return Problem(**{**d, "values": tuple(Value(**v) for v in d["values"]), "actions": tuple(d["actions"]),
                      "profiles": {k: Cost(**v) for k, v in d["profiles"].items()},
                      "precedence_edges": tuple(tuple(x) for x in d["precedence_edges"]),
                      "state_chains": tuple(tuple(x) for x in d["state_chains"]), "kv_order": tuple(d["kv_order"]),
                      "required_action_ids": tuple(d["required_action_ids"])})


def load_problem(path):
    return problem_from_dict(json.loads(Path(path).read_text()))


def validate_problem(p):
    """Validate mathematical ports and backend payload declarations, not just times."""
    require(tuple(p.kv_order) in ((0, 1), (1, 0)) and p.arena_slabs in (4, 8)
            and integer(p.register_budget_bytes, 1), "Invalid problem geometry/capacity")
    values = {v.name: v for v in p.values}
    require(all(isinstance(c, Cost) for c in p.profiles.values()), "Invalid Cost declarations")
    units = {c.unit for c in p.profiles.values()}
    require(len(units) == 1 and next(iter(units)) == p.provenance["objective_unit"], "Mixed objective units")
    require(_validate_objective_spec(p.objective_spec) == p.objective_spec and
            (p.objective_spec is None or units == {p.objective_spec["unit"]}), "Objective specification/unit changed")
    require(_validate_source_bounds(p.provenance.get("source_feature_bounds", {}), p.objective_spec) ==
            p.provenance.get("source_feature_bounds", {}), "Source bounds changed")
    require(len(values) == len(p.values), "Duplicate mathematical values")
    replication = p.provenance["backend_spec"]["register_replication_by_role"]
    expected = {"Q": ("q", None, None, 16384)}
    full_parent=p.provenance["backend_spec"].get("full_value_parent")
    require(isinstance(full_parent,bool),"V transfer-parent declaration missing")
    for j in (0, 1):
        expected.update({f"K{j}": ("k", j, None, 8192), f"S{j}": ("s", j, None, 8192),
                         f"P{j}": ("p", j, None, 4096), f"m{j}": ("m", j, None, 256),
                         f"l{j}": ("l", j, None, 256)})
        if full_parent:expected[f"V{j}"]=("v",j,None,8192)
        for stripe in (0, 1):
            expected[f"V{j}_{stripe}"] = ("v", j, stripe, 4096)
    first, second = p.kv_order
    expected["alpha1"] = ("alpha", second, None, 256)
    require(set(values) == set(expected), "Incomplete/extra mathematical inventory")
    for name, (role, tile, stripe, size) in expected.items():
        v = values[name]
        require((v.role, v.tile, v.stripe, v.logical_bytes, v.register_bytes) ==
                (role, tile, stripe, size, size*replication[role]), "Backend inventory/payload mismatch")
    require(p.provenance["backend_spec"]["row_sum_representation"] == "full_sum_quad_replicated", "Wrong row-sum representation")
    require(p.provenance["backend_spec"]["copy_semantics"] == "async_issue_ready_mask_wait" and
            p.provenance["backend_spec"]["close_workspace_smem_bytes"] in (0, 8192), "Unknown native storage contract")
    require(p.shared_alignment_slabs_by_role == p.provenance["backend_spec"]["shared_alignment_slabs_by_role"]
            and all(integer(x, 1) for x in p.shared_alignment_slabs_by_role.values()), "Backend alignment declaration mismatch")
    known = {a["id"] for a in p.actions}
    require(len(known) == len(p.actions) and set(p.required_action_ids) <= known, "Duplicate/unknown action")
    forbidden = p.provenance.get("forbidden_source_orders", [])
    require(isinstance(forbidden, list) and all(isinstance(order, list) and len(order) == len(known)
            and set(order) == known for order in forbidden)
            and len({tuple(order) for order in forbidden}) == len(forbidden), "Forbidden orders must be distinct complete source permutations")
    if forbidden:
        require({a["id"] for a in p.actions if a["mandatory"]} | set(p.required_action_ids) == known,
                "Forbidden complete orders require every study action to execute")
    mandatory = {f"stats_{first}#0", f"stats_{second}#0", "store_lse#0"}
    chains = []
    for stripe in (0, 1):
        chain = (f"pv_{first}_{stripe}#0", f"rescale_1_{stripe}#0", f"pv_{second}_{stripe}#0", f"close_{stripe}#0")
        chains.append(chain)
        mandatory.update(chain)
    require(tuple(p.state_chains) == tuple(chains), "Output state chain changed")
    require({a["id"] for a in p.actions if a["mandatory"]} == mandatory, "Exactly-once action declaration changed")
    required_edges = {(f"stats_{first}#0", f"stats_{second}#0")}
    for chain in chains:
        required_edges.update(zip(chain, chain[1:]))
    for j in (0, 1):
        if f"qk_{j}#1" in known:
            required_edges.add((f"stats_{j}#0", f"qk_{j}#1"))
    require(required_edges <= set(p.precedence_edges) and
            all(a in known and b in known and a != b for a, b in p.precedence_edges), "Required dependency edge changed")
    for a in p.actions:
        kind, j, stripe = a["kind"], a["tile"], a["stripe"]
        require(a["id"] == f"{a['name']}#{a['instance']}" and integer(a["instance"]), "Malformed instance ID")
        require(a["profile_key"] in p.profiles and a["duration"] == 1 and not isinstance(a["duration"], bool)
                and integer(a["profile_cost"]) and a["profile_cost"] == p.profiles[a["profile_key"]].duration
                and integer(a["workspace_reg_bytes"]) and integer(a["workspace_smem_bytes"]), "Action cost/workspace differs from profile")
        require(a["workspace_smem_bytes"] == (p.provenance["backend_spec"]["close_workspace_smem_bytes"] if kind == "normalize_store" else 0), "Close shared workspace contract changed")
        ins, outs = [tuple(x) for x in a["input_refs"]], [tuple(x) for x in a["output_refs"]]
        require(all(name in values and tier in ("gmem", "smem", "reg") for name, tier in [*ins, *outs]), "Unknown mathematical reference")
        if kind in ("copy_g2s", "load_s2r", "store_r2s"):
            require(len(outs) == 1 and outs[0][0] in values, "Bad transfer target")
            v = values[outs[0][0]]
            source_tier, dest_tier = {"copy_g2s": ("gmem", "smem"), "load_s2r": ("smem", "reg"), "store_r2s": ("reg", "smem")}[kind]
            source_name=f"V{v.tile}" if full_parent and kind=="load_s2r" and v.role=="v" else v.name
            require(ins==[(source_name,source_tier)] and outs==[(v.name,dest_tier)] and j==v.tile and stripe==v.stripe,"Transfer math/layout ports changed")
            if full_parent and v.role=="v":
                require((kind=="copy_g2s" and v.stripe is None) or (kind=="load_s2r" and v.stripe in (0,1)),
                    "Parent V transfers must copy whole128 columns and load separate64-column RF views")
            require(v.role in (("q", "k", "v") if kind == "copy_g2s" else ("s", "p") if kind == "store_r2s" else ("q", "k", "v", "s", "p")), "Illegal transfer role")
            continue
        require(a["instance"] < (2 if kind == "qk" else 1), "Producer bound exceeded")
        if kind == "qk":
            expected_ins, expected_outs = [("Q", "reg"), (f"K{j}", "reg")], [(f"S{j}", "reg")]
        elif kind == "stats_first":
            require(j == first and a["first"], "Wrong first prefix tile")
            expected_ins, expected_outs = [(f"S{j}", "reg")], [(f"m{j}", "reg"), (f"l{j}", "reg"), (f"P{j}", "reg")]
        elif kind == "stats_next":
            require(j == second and not a["first"], "Wrong second prefix tile")
            expected_ins = [(f"S{j}", "reg"), (f"m{first}", "reg"), (f"l{first}", "reg")]
            expected_outs = [(f"m{j}", "reg"), (f"l{j}", "reg"), ("alpha1", "reg"), (f"P{j}", "reg")]
        elif kind == "recover_p":
            expected_ins, expected_outs = [(f"S{j}", "reg"), (f"m{j}", "reg")], [(f"P{j}", "reg")]
        elif kind in ("pv_first", "pv_next"):
            require(stripe in (0, 1) and j == (first if kind == "pv_first" else second), "Wrong PV tile/stripe")
            expected_ins, expected_outs = [(f"P{j}", "reg"), (f"V{j}_{stripe}", "reg")], []
        elif kind == "rescale_o":
            require(stripe in (0, 1) and j == second, "Wrong rescale stripe")
            expected_ins, expected_outs = [("alpha1", "reg")], []
        elif kind == "normalize_store":
            require(stripe in (0, 1), "Wrong close stripe")
            expected_ins, expected_outs = [(f"l{second}", "reg")], []
        elif kind == "store_lse":
            expected_ins, expected_outs = [(f"m{second}", "reg"), (f"l{second}", "reg")], []
        else:
            raise ValueError("Unknown mathematical action")
        require(ins == expected_ins and outs == expected_outs, "Mathematical action ports changed")
    return True


def solve(p: Problem, *, max_time_seconds=30.0, artifact_dir=None, initial_hint=None):
    from ortools.sat.python import cp_model
    validate_problem(p)
    require(max_time_seconds > 0, "Positive time limit required")
    values = {v.name: v for v in p.values}
    values.update({_workspace_value(a).name: _workspace_value(a) for a in p.actions if a["workspace_smem_bytes"]})
    horizon = len(p.actions)
    model = cp_model.CpModel()
    finish = model.new_int_var(1, horizon, "finish")
    av, copies, bindings = {}, [], []
    rf_intervals, rf_sizes, engine, rectangles, rf_spans = [], [], [], [], []
    for a in p.actions:
        aid = a["id"]
        use = model.new_bool_var("use:" + aid)
        start = model.new_int_var(0, horizon, "start:" + aid)
        end = model.new_int_var(0, horizon, "end:" + aid)
        interval = model.new_optional_interval_var(start, a["duration"], end, use, aid)
        model.add(start == 0).only_enforce_if(use.Not())
        model.add(end == 0).only_enforce_if(use.Not())
        model.add(end <= finish)
        if a["mandatory"] or aid in p.required_action_ids:
            model.add(use == 1)
        if a["instance"]:
            previous = av[f"{a['name']}#{a['instance']-1}"]
            model.add(use <= previous["use"])
            model.add(start >= previous["start"]).only_enforce_if(use)
        av[aid] = dict(use=use, start=start, end=end, interval=interval)
        engine.append(interval)
        if a["workspace_reg_bytes"]:
            rf_intervals.append(interval)
            rf_sizes.append(a["workspace_reg_bytes"])
            rf_spans.append((start, end, use, a["workspace_reg_bytes"]))
    model.add_no_overlap(engine)
    model.add_max_equality(finish, [v["end"] for v in av.values()])
    # No idle clock time exists in an ordinal program. Any source order can
    # be packed without changing bindings/liveness overlap. This also removes
    # CP-SAT symmetry among arbitrarily gapped representations of one order.
    model.add(finish == sum(v["use"] for v in av.values()))
    for index, order in enumerate(p.provenance.get("forbidden_source_orders", [])):
        reversals = []
        for left, right in zip(order, order[1:]):
            reversed_pair = model.new_bool_var(f"forbidden:{index}:{left}:{right}")
            model.add(av[left]["start"] >= av[right]["end"]).only_enforce_if(reversed_pair)
            reversals.append(reversed_pair)
        model.add(sum(reversals) >= 1)
    for before, after in p.precedence_edges:
        model.add(av[before]["end"] <= av[after]["start"]).only_enforce_if([av[before]["use"], av[after]["use"]])
    for action in p.actions:
        if action["kind"] != "recover_p":
            continue
        recovery = av[action["id"]]
        predecessors = []
        for pv in p.actions:
            if pv["kind"] in ("pv_first", "pv_next") and pv["tile"] == action["tile"]:
                before = model.new_bool_var("recovery_after:"+pv["id"]+":"+action["id"])
                model.add(before <= recovery["use"])
                model.add(av[pv["id"]]["end"] <= recovery["start"]).only_enforce_if(before)
                predecessors.append(before)
        model.add(sum(predecessors) >= recovery["use"])
    state_vars = []
    for stripe, chain in enumerate(p.state_chains):
        start, end = av[chain[0]]["start"], av[chain[-1]]["end"]
        length = model.new_int_var(0, horizon, f"O{stripe}:life")
        model.add(length == end - start)
        interval = model.new_interval_var(start, length, end, f"mutable O{stripe}")
        rf_intervals.append(interval)
        rf_sizes.append(OUTPUT_STRIPE_BYTES)
        rf_spans.append((start, end, model.new_constant(1), OUTPUT_STRIPE_BYTES))
        state_vars.append((stripe, chain, start, end))

    def allocate(identity, ref, start, ready, minimum_end, use, producer=None, workspace=False):
        name, tier = ref
        end = model.new_int_var(0, horizon, "free:" + identity)
        length = model.new_int_var(0, horizon, "life:" + identity)
        model.add(length == end - start)
        interval = model.new_optional_interval_var(start, length, end, use, "life:" + identity)
        size = values[name].logical_bytes if tier == "gmem" else _occupied(values[name], tier)
        c = dict(id=identity, ref=tuple(ref), start=start, available=ready, end=end, minimum_end=minimum_end,
                 use=use, producer=producer, uses=[], occupied_bytes=size, offset=None, workspace=workspace, length=length)
        if tier == "gmem":
            model.add(end == finish)
        elif tier == "reg":
            rf_intervals.append(interval)
            rf_sizes.append(size)
            rf_spans.append((start, end, use, size))
        else:
            slots = size // SLAB_BYTES
            alignment = 1 if workspace else p.shared_alignment_slabs_by_role[values[name].role]
            if slots > p.arena_slabs:
                model.add(use == 0)
                offset = model.new_int_var(0, 0, "offset:" + identity)
            else:
                offset = model.new_int_var_from_domain(cp_model.Domain.from_values(range(0, p.arena_slabs-slots+1, alignment)), "offset:" + identity)
            high = model.new_int_var(0, max(p.arena_slabs, slots), "high:" + identity)
            model.add(high == offset + slots)
            address = model.new_optional_interval_var(offset, slots, high, use, "address:" + identity)
            rectangles.append((interval, address))
            c["offset"] = offset
        copies.append(c)
        if workspace:
            model.add(end == minimum_end)

    for value in p.values:
        if value.role in ("q", "k", "v") and not (p.provenance["backend_spec"]["full_value_parent"] and value.role=="v" and value.stripe is not None):
            allocate("gmem:" + value.name, (value.name, "gmem"), 0, 0, 0, model.new_constant(1))
    for a in p.actions:
        v = av[a["id"]]
        for port, ref in enumerate(a["output_refs"]):
            allocate(f"{a['id']}:out{port}", ref, v["start"], v["end"], v["end"], v["use"], a["id"])
        if a["workspace_smem_bytes"]:
            allocate("workspace:"+a["id"], (_workspace_value(a).name, "smem"),
                     v["start"], v["end"], v["end"], v["use"], a["id"], workspace=True)
    for a in p.actions:
        consumer = av[a["id"]]
        for port, ref in enumerate(a["input_refs"]):
            choices = []
            for c in copies:
                if c["ref"] != tuple(ref) or c["producer"] == a["id"]:
                    continue
                use = model.new_bool_var(f"bind:{a['id']}:{port}:{c['id']}")
                model.add(use <= consumer["use"])
                model.add(use <= c["use"])
                model.add(c["available"] <= consumer["start"]).only_enforce_if(use)
                model.add(c["end"] >= consumer["end"]).only_enforce_if(use)
                actual_end = model.new_int_var(0, horizon, "used:" + str(len(bindings)))
                model.add(actual_end == consumer["end"]).only_enforce_if(use)
                model.add(actual_end == 0).only_enforce_if(use.Not())
                c["uses"].append((use, actual_end))
                bindings.append(dict(consumer=a["id"], port=port, copy=c["id"], use=use))
                choices.append(use)
            model.add(sum(choices) == consumer["use"])
    for c in copies:
        if c["producer"] and not c["workspace"]:
            # Every materialized output has a genuine future consumer. In
            # particular stats' initial P cannot be a discarded unpriced cast.
            model.add(sum(x[0] for x in c["uses"]) >= c["use"])
            model.add_max_equality(c["end"], [c["minimum_end"], *[x[1] for x in c["uses"]]])
    if rectangles:
        model.add_no_overlap_2d([r[0] for r in rectangles], [r[1] for r in rectangles])
    model.add_cumulative(rf_intervals, rf_sizes, p.register_budget_bytes)
    source_cache = {}
    copy_by_id = {c["id"]: c for c in copies}
    if "residency_exchange_v78" in p.provenance:
        from .exchange import constrain
        constrain(model, p.provenance["residency_exchange_v78"], av, copies, bindings)
    action_by_id = {a["id"]: a for a in p.actions}
    frontier_cp = {}
    def exact_frontier_cp():
        if frontier_cp:
            return frontier_cp
        require(horizon==_expected_steps(p) and all(a["instance"] == 0 and a["kind"] in ACTION_KINDS for a in p.actions)
                and {a["id"] for a in p.actions if a["mandatory"]} | set(p.required_action_ids) == set(av)
                and p.provenance["execution_mode"] == "source_order"
                and p.provenance["backend_spec"]["close_workspace_smem_bytes"] == 8192,
                "Exact CP frontier requires split27 or parent25 mandatory shared128 actions, no checkpoints/recovery/guards")
        require(p.arena_slabs in (4,8),"V52 exact mask scope is four or eight4KiB units")
        # Every distributed read remains in the slab union until a genuine CTA
        # barrier. A ready-aware load may accumulate a second disjoint read.
        masks=list(range(1 << p.arena_slabs))
        smem_spans = {}
        for c in copies:
            if c["ref"][1] != "smem":
                continue
            slots = c["occupied_bytes"]//SLAB_BYTES
            require(slots in (1, 2, 4), "Unsupported lowering-frontier span width")
            selected = model.new_int_var(1, masks[-1], "frontier_mask:"+c["id"])
            model.add_allowed_assignments([c["offset"], selected],
                [(offset, ((1<<slots)-1)<<offset) for offset in range(p.arena_slabs-slots+1)])
            smem_spans[c["id"]] = selected
        modes, read_spans, write_spans, classes, producer_ends = [], [], [], [], []
        for a in p.actions:
            outputs = [c for c in copies if c["producer"] == a["id"] and c["ref"][1] == "smem"]
            if a["kind"] == "copy_g2s":
                require(len(outputs) == 1, "G2S needs exactly one physical output")
                modes.append(2); read_spans.append(0); write_spans.append(smem_spans[outputs[0]["id"]])
                classes.append({"q": 1, "k": 2, "v": 3}[values[outputs[0]["ref"][0]].role])
                producer_ends.append(0)
            elif a["kind"] == "normalize_store":
                require(len(outputs) == 1 and outputs[0]["workspace"], "Close needs one physical scratch")
                modes.append(3); read_spans.append(smem_spans[outputs[0]["id"]]); write_spans.append(smem_spans[outputs[0]["id"]]); classes.append(4)
                producer_ends.append(0)
            elif a["kind"] == "load_s2r":
                sources = [c for c in copies if c["ref"] == tuple(a["input_refs"][0])]
                require(len(sources) == 1 and values[sources[0]["ref"][0]].role in ("q", "k", "v"), "Only unique immutable Q/K/V loads are in frontier scope")
                source=sources[0];read_mask=smem_spans[source['id']]
                if values[source['ref'][0]].role=='v' and values[source['ref'][0]].stripe is None:
                    stripe=values[a['output_refs'][0][0]].stripe
                    require(stripe in (0,1),'Invalid full parent view')
                    read_mask=model.new_int_var(1,masks[-1],'parent_read_mask:'+a['id'])
                    model.add_allowed_assignments([source['offset'],read_mask],
                        [(offset,1<<(offset+stripe)) for offset in range(p.arena_slabs-2+1)])
                modes.append(1); read_spans.append(read_mask); write_spans.append(0); classes.append(0)
                producer_ends.append(av[sources[0]["producer"]]["end"])
            else:
                modes.append(0); read_spans.append(0); write_spans.append(0); classes.append(0)
                producer_ends.append(0)
        positions = [av[a["id"]]["start"] for a in p.actions]
        at_step = [model.new_int_var(0, horizon-1, f"action_at:{step}") for step in range(horizon)]
        slots = []
        for index, position in enumerate(positions):
            model.add(position < horizon)
            row = []
            for step in range(horizon):
                selected = model.new_bool_var(f"action_slot:{index}:{step}")
                model.add(position == step).only_enforce_if(selected)
                model.add(position != step).only_enforce_if(selected.Not())
                row.append(selected)
            model.add(sum(row) == 1)
            slots.append(row)
        for step in range(horizon):
            model.add(sum(row[step] for row in slots) == 1)
            model.add(at_step[step] == sum(index*slots[index][step] for index in range(horizon)))
        states = [model.new_int_var(0, masks[-1], f"pending_read_mask:{step}") for step in range(horizon+1)]
        model.add(states[0] == 0)
        ready = [model.new_int_var(0,horizon,f"ready_cutoff:{step}") for step in range(horizon+1)]
        model.add(ready[0]==0)
        transitions = []
        physical_masks=sorted({((1<<size)-1)<<offset for size in (1,2,4) for offset in range(p.arena_slabs-size+1)})
        for before in masks:
            transitions.append((before,0,0,0,0,before,0))
            for address in physical_masks:
                transitions.append((before,1,address,0,0,before|address,0))
                transitions.append((before,1,address,0,1,address,0))
                hazard=int(bool(before&address))
                transitions.append((before,2,0,address,0,0 if hazard else before,hazard))
                transitions.append((before,3,address,address,0,address,hazard))
        releases, release_classes, waits = [], [], []
        math_weights=[[MATH_SITE_WEIGHTS.get(a['kind'],(0,0))[kind] for a in p.actions] for kind in (0,1)]
        math_maxima=[sum(weights) for weights in math_weights]
        math_gaps=[[model.new_int_var(0,maximum,f'last_commit_math:{kind}:{step}') for step in range(horizon+1)]
                   for kind,maximum in enumerate(math_maxima)]
        math_waits=[[],[]]
        for gaps in math_gaps:model.add(gaps[0]==0)
        for step in range(horizon):
            mode = model.new_int_var(0, 3, f"mode_at:{step}")
            read = model.new_int_var(0, masks[-1], f"read_at:{step}")
            write = model.new_int_var(0, masks[-1], f"write_at:{step}")
            role = model.new_int_var(0, 4, f"write_class_at:{step}")
            for array, target in ((modes, mode), (read_spans, read), (write_spans, write), (classes, role)):
                model.add_element(at_step[step], array, target)
            producer_end=model.new_int_var(0,horizon,f"producer_end_at:{step}")
            model.add_element(at_step[step],producer_ends,producer_end)
            wait=model.new_bool_var(f"wait0_at:{step}")
            # Non-load actions have producer_end0 and can never need a wait.
            model.add(ready[step]<producer_end).only_enforce_if(wait)
            model.add(ready[step]>=producer_end).only_enforce_if(wait.Not())
            model.add(ready[step+1]==step+1).only_enforce_if(wait)
            model.add(ready[step+1]==ready[step]).only_enforce_if(wait.Not())
            release = model.new_bool_var(f"release_at:{step}")
            model.add_allowed_assignments([states[step],mode,read,write,wait,states[step+1],release], transitions)
            copy=model.new_bool_var(f'copy_at:{step}')
            model.add(mode==2).only_enforce_if(copy)
            model.add(mode!=2).only_enforce_if(copy.Not())
            reset=model.new_bool_var(f'math_gap_reset:{step}')
            model.add_max_equality(reset,[copy,wait])
            for kind,(weights,maximum,gaps) in enumerate(zip(math_weights,math_maxima,math_gaps)):
                work=model.new_int_var(0,max(weights),f'math_work:{kind}:{step}')
                model.add_element(at_step[step],weights,work)
                model.add(gaps[step+1]==0).only_enforce_if(reset)
                model.add(gaps[step+1]==gaps[step]+work).only_enforce_if(reset.Not())
                counted=model.new_int_var(0,maximum,f'wait_math:{kind}:{step}')
                model.add(counted==gaps[step]).only_enforce_if(wait)
                model.add(counted==0).only_enforce_if(wait.Not())
                math_waits[kind].append(counted)
            release_class = model.new_int_var(0, 4, f"release_class_at:{step}")
            model.add(release_class == 0).only_enforce_if(release.Not())
            model.add(release_class == role).only_enforce_if(release)
            releases.append(release); release_classes.append(release_class);waits.append(wait)
        frontier_cp.update(masks=masks, states=states, ready=ready, waits=waits,
                          releases=releases, release_classes=release_classes,
                          math_gaps=math_gaps,math_waits=math_waits,math_maxima=math_maxima)
        return frontier_cp
    def source_cp(source):
        """Exact bounded ordinal summaries, independently extracted in replay."""
        key = json.dumps(source, sort_keys=True)
        if key in source_cache:
            return source_cache[key]
        kind, roles = source["kind"], source.get("roles", VALUE_ROLES)
        def kib(size):
            require(size % 1024 == 0, "KiB-step source features require exact KiB payloads")
            return size // 1024
        label = "source:" + str(len(source_cache)) + ":" + kind
        if kind == "consumer_wait_count":
            frontier=exact_frontier_cp()
            maximum=horizon
            value=model.new_int_var(0,maximum,label)
            model.add(value==sum(frontier["waits"]))
        elif kind in ('last_commit_wait_hmma_sites','last_commit_wait_ex2_sites'):
            frontier=exact_frontier_cp()
            index=0 if kind=='last_commit_wait_hmma_sites' else 1
            maximum=frontier['math_maxima'][index]
            value=model.new_int_var(0,maximum,label)
            model.add(value==sum(frontier['math_waits'][index]))
        elif kind == "release_barrier_count":
            frontier = exact_frontier_cp()
            allowed = set()
            for category, write_kind, role in ((1, "copy_g2s", "q"), (2, "copy_g2s", "k"),
                                               (3, "copy_g2s", "v"), (4, "normalize_store", "workspace")):
                if write_kind in source.get("write_kinds", ("copy_g2s", "normalize_store")) and ("roles" not in source or role in roles):
                    allowed.add(category)
            counters = []
            for step, category in enumerate(frontier["release_classes"]):
                counted = model.new_bool_var(f"{label}:counted:{step}")
                model.add_allowed_assignments([category, counted], [(c, int(c in allowed)) for c in range(5)])
                counters.append(counted)
            maximum = horizon
            value = model.new_int_var(0, maximum, label)
            model.add(value == sum(counters))
        elif kind in ("rf_kib_steps", "rf_hold_kib_steps", "o_kib_steps"):
            parts, maximum = [], 0
            if kind != "o_kib_steps":
                for c in copies:
                    if c["ref"][1] == "reg" and (kind == "rf_kib_steps" or values[c["ref"][0]].role in roles):
                        size = kib(c["occupied_bytes"])
                        begin = c["start"] if kind == "rf_kib_steps" else c["available"]
                        parts.append(size*(c["end"]-begin))
                        maximum += size*horizon
            if kind in ("rf_kib_steps", "o_kib_steps"):
                parts.extend(kib(OUTPUT_STRIPE_BYTES)*(end-start) for _, _, start, end in state_vars)
                maximum += len(state_vars)*kib(OUTPUT_STRIPE_BYTES)*horizon
            if kind == "rf_kib_steps":
                parts.extend(kib(a["workspace_reg_bytes"])*av[a["id"]]["use"] for a in p.actions)
                maximum += sum(kib(a["workspace_reg_bytes"]) for a in p.actions)
            value = model.new_int_var(0, maximum, label)
            model.add(value == sum(parts))
        elif kind == "o_overlap_steps":
            start = model.new_int_var(0, horizon, label+":start")
            end = model.new_int_var(0, horizon, label+":end")
            model.add_max_equality(start, [x[2] for x in state_vars])
            model.add_min_equality(end, [x[3] for x in state_vars])
            value, maximum = model.new_int_var(0, horizon, label), horizon
            model.add_max_equality(value, [0, end-start])
        elif kind in ("g2s_to_s2r_gap_steps", "dependency_gap_steps", "g2s_interposed_actions"):
            gaps = []
            for b in bindings:
                c, consumer = copy_by_id[b["copy"]], action_by_id[b["consumer"]]
                producer = action_by_id.get(c["producer"])
                if producer is None or values[c["ref"][0]].role not in roles:
                    continue
                if kind in ("g2s_to_s2r_gap_steps", "g2s_interposed_actions"):
                    include = producer["kind"] == "copy_g2s" and consumer["kind"] == "load_s2r"
                else:
                    include = c["ref"][1] in source.get("tiers", ("reg", "smem"))
                if include:
                    if kind == "g2s_interposed_actions":
                        for middle in p.actions:
                            if middle["kind"] not in source.get("kinds", ACTION_KINDS):
                                continue
                            action = av[middle["id"]]
                            after = model.new_bool_var(label+":after:"+str(len(gaps)))
                            before = model.new_bool_var(label+":before:"+str(len(gaps)))
                            model.add(action["start"] >= c["available"]).only_enforce_if(after)
                            model.add(action["start"] < c["available"]).only_enforce_if(after.Not())
                            model.add(action["end"] <= av[b["consumer"]]["start"]).only_enforce_if(before)
                            model.add(action["end"] > av[b["consumer"]]["start"]).only_enforce_if(before.Not())
                            counted = model.new_bool_var(label+":interposed:"+str(len(gaps)))
                            for condition in (b["use"], action["use"], after, before):
                                model.add(counted <= condition)
                            model.add(counted >= b["use"]+action["use"]+after+before-3)
                            gaps.append(counted)
                        continue
                    gap = model.new_int_var(0, horizon, label+":gap"+str(len(gaps)))
                    model.add(gap == av[b["consumer"]]["start"]-c["available"]).only_enforce_if(b["use"])
                    model.add(gap == 0).only_enforce_if(b["use"].Not())
                    gaps.append(gap)
            maximum = len(gaps)*horizon
            value = model.new_int_var(0, maximum, label)
            model.add(value == sum(gaps))
        elif kind == "rf_peak_kib":
            loads = []
            maximum = p.register_budget_bytes // 1024
            for step in range(horizon):
                active_terms = []
                for i, (start, end, use, size) in enumerate(rf_spans):
                    started = model.new_bool_var(f"{label}:started:{step}:{i}")
                    retained = model.new_bool_var(f"{label}:retained:{step}:{i}")
                    active = model.new_bool_var(f"{label}:active:{step}:{i}")
                    model.add(start <= step).only_enforce_if(started)
                    model.add(start > step).only_enforce_if(started.Not())
                    model.add(end > step).only_enforce_if(retained)
                    model.add(end <= step).only_enforce_if(retained.Not())
                    model.add(active <= use)
                    model.add(active <= started)
                    model.add(active <= retained)
                    model.add(active >= use+started+retained-2)
                    active_terms.append(kib(size)*active)
                load = model.new_int_var(0, maximum, f"{label}:load:{step}")
                model.add(load == sum(active_terms))
                loads.append(load)
            value = model.new_int_var(0, maximum, label)
            model.add_max_equality(value, loads)
        else:
            raise ValueError("Unsupported CP source feature")
        source_cache[key] = (value, maximum)
        return value, maximum
    for kind, (minimum, maximum) in p.provenance.get("source_feature_bounds", {}).items():
        value, _ = source_cp(source_definition(kind))
        model.add(value >= minimum)
        model.add(value <= maximum)
    if p.objective_spec is None:
        coefficients = {a["id"]: a["profile_cost"] for a in p.actions}
        constant = 0
    else:
        constant = p.objective_spec["constant"]
        coefficients = {a["id"]: sum(t["coefficient"] for t in p.objective_spec["terms"]
                                     if "match" in t and action_matches(a, t["match"], values)) for a in p.actions}
    source_terms = []
    for term in p.objective_spec["terms"] if p.objective_spec else ():
        if "source" in term:
            feature, maximum = source_cp(term["source"])
            source_terms.append((term["coefficient"], feature, maximum))
    lower = constant + sum(min(0, c) for c in coefficients.values()) + sum(min(0, c*maximum) for c, _, maximum in source_terms)
    upper = constant + sum(max(0, c) for c in coefficients.values()) + sum(max(0, c*maximum) for c, _, maximum in source_terms)
    require(max(abs(lower), abs(upper))*(horizon+1)+horizon < 2**62, "Objective scaling exceeds safe integer range")
    cost = model.new_int_var(lower, upper, "objective_proxy")
    model.add(cost == constant + sum(coefficients[a["id"]]*av[a["id"]]["use"] for a in p.actions)
              + sum(coefficient*feature for coefficient, feature, _ in source_terms))
    # Source steps are merely the second, canonicalization objective. A change
    # in fitted coefficients can select another ORDER, but cannot stretch a
    # primitive interval or pretend that register data lived hardware ns.
    model.minimize(cost*(horizon+1) + finish)
    if initial_hint is not None:
        require(initial_hint.get('status') in ('FEASIBLE','OPTIMAL'),'A source hint requires a retained feasible plan')
        hint_events={event['id']:event for event in initial_hint['events']}
        require(set(hint_events)==set(av),'Hint action vocabulary differs')
        for identity,variables in av.items():model.add_hint(variables['start'],hint_events[identity]['start'])
        hint_copies={copy['id']:copy for copy in initial_hint['copies']}
        for copy in copies:
            if copy['offset'] is not None and copy['id'] in hint_copies:
                offset=hint_copies[copy['id']]['offset_bytes']
                require(offset%SLAB_BYTES==0,'Source hint offset is not4KiB aligned')
                model.add_hint(copy['offset'],offset//SLAB_BYTES)
    text = str(model.proto)
    artifacts = Path(artifact_dir) if artifact_dir else None
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=False)
        (artifacts / "problem.json").write_text(json.dumps(asdict(p), indent=2) + "\n")
        (artifacts / "model.textproto").write_text(text)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    before = time.perf_counter()
    status = solver.solve(model)
    r = dict(status=solver.status_name(status), ordering_unit="source_steps", time_unit="source_steps",
             objective_unit=p.provenance["objective_unit"], objective=None, best_bound=None, finish_step=None,
             optimality_certified=status == cp_model.OPTIMAL, events=[], copies=[], output_states=[],
             profiles={k: asdict(c) for k, c in p.profiles.items()}, provenance=dict(p.provenance),
             model_sha256=hashlib.sha256(text.encode()).hexdigest(),
             ortools_version=importlib.metadata.version("ortools"), solver_wall_seconds=time.perf_counter()-before)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        r["finish_step"] = solver.value(finish)
        r["objective"] = solver.value(cost)
        r["best_bound"] = r["objective"] if status == cp_model.OPTIMAL else math.floor(solver.best_objective_bound/(horizon+1)+1e-9)
        r["lexicographic_objective"] = {"combined": r["objective"]*(horizon+1)+r["finish_step"],
                                        "combined_bound": solver.best_objective_bound, "cost_multiplier": horizon+1}
        selected = [b for b in bindings if solver.value(b["use"])]
        for c in copies:
            if not solver.value(c["use"]):
                continue
            value = values[c["ref"][0]]
            r["copies"].append(dict(id=c["id"], value=value.name, role=value.role, tile=value.tile, stripe=value.stripe,
                tier=c["ref"][1], logical_bytes=value.logical_bytes, nbytes=value.logical_bytes,
                occupied_bytes=c["occupied_bytes"], layout=value.layout, producer=c["producer"],
                start=c["start"] if isinstance(c["start"], int) else solver.value(c["start"]),
                available=c["available"] if isinstance(c["available"], int) else solver.value(c["available"]),
                end=solver.value(c["end"]), offset_bytes=None if c["offset"] is None else solver.value(c["offset"])*SLAB_BYTES,
                workspace=c["workspace"]))
        for a in p.actions:
            if not solver.value(av[a["id"]]["use"]):
                continue
            e = {k: a[k] for k in ("id", "name", "kind", "instance", "value", "tile", "stripe", "first", "profile_key", "duration", "profile_cost", "workspace_reg_bytes", "workspace_smem_bytes")}
            e.update(start=solver.value(av[a["id"]]["start"]), end=solver.value(av[a["id"]]["end"]),
                     inputs=[b["copy"] for b in sorted(selected, key=lambda b: b["port"]) if b["consumer"] == a["id"]],
                     outputs=[c["id"] for c in r["copies"] if c["producer"] == a["id"] and not c["workspace"]],
                     workspace_copies=[c["id"] for c in r["copies"] if c["producer"] == a["id"] and c["workspace"]],
                     state_effect=any(a["id"] in chain for chain in p.state_chains))
            r["events"].append(e)
        r["events"].sort(key=lambda e: (e["start"], e["id"]))
        r["actions"] = r["events"]
        r["output_states"] = [dict(stripe=s, tier="reg", bytes=OUTPUT_STRIPE_BYTES,
            start=solver.value(start), end=solver.value(end), first_event=chain[0], last_event=chain[-1],
            effect_order=list(chain)) for s, chain, start, end in state_vars]
        r["source_features"] = source_feature_values(p, r)
        r["lowering_trace"] = lowering_trace(p, r)
        r["cp_lowering_frontier"] = None
        if frontier_cp:
            certificate = []
            for step, row in enumerate(r["lowering_trace"]):
                require(solver.value(frontier_cp["states"][step])==row["read_mask_before"]
                        and solver.value(frontier_cp["states"][step+1])==row["read_mask_after"]
                        and solver.value(frontier_cp["ready"][step])==row["ready_cutoff_before"]
                        and solver.value(frontier_cp["ready"][step+1])==row["ready_cutoff_after"]
                        and solver.value(frontier_cp["waits"][step])==row["wait_before_load"]
                        and solver.value(frontier_cp["releases"][step])==row["release_before_write"],
                        "CP ready/read-mask frontier differs from independent replay")
                for k,metric in enumerate(('hmma','ex2')):
                    require(solver.value(frontier_cp['math_gaps'][k][step])==row['last_commit_'+metric+'_before']
                        and solver.value(frontier_cp['math_gaps'][k][step+1])==row['last_commit_'+metric+'_after']
                        and solver.value(frontier_cp['math_waits'][k][step])==row['last_commit_wait_'+metric+'_sites'],
                        'CP last-commit overlap differs from independent replay')
                certificate.append({key:row[key] for key in CP_CERTIFICATE_KEYS})
            r["cp_lowering_frontier"]={"kind":"exact-fourk-mask-ready-cutoff-last-commit-overlap", "slab_bytes":SLAB_BYTES,
                                      "arena_slabs":p.arena_slabs,"events":certificate}
        r["objective_features"] = objective_features(p, r["events"], r)
        r["cp_source_features"] = {key: solver.value(value) for key, (value, _) in source_cache.items()}
        require(all(value == _source_value(p, r, json.loads(key)) for key, value in r["cp_source_features"].items()),
                "CP source summaries differ from independent allocation replay")
        r["objective_model"] = {"kind": "affine_features" if p.objective_spec else "action_coefficient_sum",
                                  "spec": p.objective_spec, "unit": p.provenance["objective_unit"]}
        r["replay"] = replay(p, r)
    if artifacts:
        (artifacts / "solver_response.textproto").write_text(str(solver.response_proto))
        (artifacts / "solver_stats.txt").write_text(solver.response_stats()+"\n")
        (artifacts / "result.json").write_text(json.dumps(r, indent=2)+"\n")
    return r


def replay(p: Problem, r):
    """Independent checks for every dependency, output state and physical slot."""
    validate_problem(p)
    desc = {a["id"]: a for a in p.actions}
    values = {v.name: v for v in p.values}
    values.update({_workspace_value(a).name: _workspace_value(a) for a in p.actions if a["workspace_smem_bytes"]})
    events = {e["id"]: e for e in r["events"]}
    copies = {c["id"]: c for c in r["copies"]}
    require(len(events) == len(r["events"]) and len(copies) == len(r["copies"]), "Duplicate event/copy")
    finish = r["finish_step"]
    require(integer(finish, 1) and r["ordering_unit"] == r["time_unit"] == "source_steps", "Invalid source-step timeline")
    require(isinstance(r["objective"], int) and not isinstance(r["objective"], bool)
            and r["objective_unit"] == p.provenance["objective_unit"], "Invalid proxy objective/unit")
    require(r["profiles"] == {key: asdict(c) for key, c in p.profiles.items()} and r["provenance"] == dict(p.provenance), "Objective/provenance declarations changed")
    require(r.get("actions") == r["events"], "actions alias differs")
    require(isinstance(r["best_bound"], int) and not isinstance(r["best_bound"], bool) and isinstance(r["optimality_certified"], bool) and r["best_bound"] <= r["objective"],
            "Invalid bound")
    require(not r["optimality_certified"] or r["best_bound"] == r["objective"], "Unproved proxy optimum")
    lex = r["lexicographic_objective"]
    require(lex["cost_multiplier"] == len(p.actions)+1 and
            lex["combined"] == r["objective"]*(len(p.actions)+1)+finish and
            isinstance(lex["combined_bound"], (int, float)) and math.isfinite(lex["combined_bound"]) and
            lex["combined_bound"] <= lex["combined"]+1e-6, "Invalid lexicographic objective record")
    require(not r["optimality_certified"] or abs(lex["combined_bound"]-lex["combined"]) <= 1e-6, "Unproved lexicographic optimum")
    require(all(a["id"] in events for a in p.actions if a["mandatory"]) and set(p.required_action_ids) <= events.keys(),
            "Required action missing")
    uses = {identity: [] for identity in copies}
    ordered = sorted(events.values(), key=lambda e: (e["start"], e["id"]))
    require([e["id"] for e in ordered] not in p.provenance.get("forbidden_source_orders", []),
            "A forbidden complete source order was selected")
    last = 0
    contract = ("name", "kind", "instance", "value", "tile", "stripe", "first", "profile_key", "duration", "profile_cost", "workspace_reg_bytes", "workspace_smem_bytes")
    for e in ordered:
        require(e["id"] in desc, "Unknown action")
        a = desc[e["id"]]
        require(all(e[k] == a[k] for k in contract), "Action contract changed")
        require(integer(e["start"]) and integer(e["end"], 1) and e["start"] >= last and
                e["start"] == last and e["end"] == e["start"] + a["duration"] <= finish, "Invalid serial source timing")
        last = e["end"]
        require(e["state_effect"] == any(a["id"] in chain for chain in p.state_chains) and isinstance(e["state_effect"], bool), "State effect changed")
        require(len(e["inputs"]) == len(a["input_refs"]) and len(e["outputs"]) == len(a["output_refs"]), "Action port mismatch")
        for identity, ref in zip(e["inputs"], a["input_refs"]):
            require(identity in copies, "Missing input copy")
            c = copies[identity]
            require((c["value"], c["tier"]) == tuple(ref) and c["available"] <= e["start"] and c["end"] >= e["end"], "Unavailable source or early release")
            uses[identity].append(e["end"])
        for port, (identity, ref) in enumerate(zip(e["outputs"], a["output_refs"])):
            require(identity == f"{e['id']}:out{port}" and identity in copies, "Output identity changed")
            c = copies[identity]
            require((c["value"], c["tier"]) == tuple(ref) and c["producer"] == e["id"] and c["start"] == e["start"] and c["available"] == e["end"], "Output provenance changed")
        require(e["workspace_copies"] == (["workspace:"+e["id"]] if a["workspace_smem_bytes"] else []), "Missing/invented close workspace")
        for identity in e["workspace_copies"]:
            require(identity in copies, "Missing scratch allocation")
            c = copies[identity]
            require(c["workspace"] is True and c["producer"] == e["id"] and c["value"] == _workspace_value(a).name
                    and c["tier"] == "smem" and c["start"] == e["start"] and c["available"] == c["end"] == e["end"], "Scratch lifetime or ownership changed")
        if a["instance"]:
            prior = events.get(f"{a['name']}#{a['instance']-1}")
            require(prior is not None and prior["start"] <= e["start"], "Instance prefix/order changed")
    require(last == finish, "Unexplained terminal slack")
    derived_features = source_feature_values(p, r)
    require(r["source_features"] == (derived_features if p.provenance["model_revision"] == MODEL_REVISION else
                                    {key: derived_features[key] for key in LEGACY_SOURCE_KEYS}), "Source-order feature record changed")
    if p.provenance["model_revision"] == MODEL_REVISION:
        trace = lowering_trace(p, r)
        require(r["lowering_trace"] == trace, "Lowering frontier replay record changed")
        frontier_features=('release_barrier_count','consumer_wait_count','last_commit_wait_hmma_sites','last_commit_wait_ex2_sites')
        requested = any(t.get("source", {}).get("kind") in frontier_features for t in p.objective_spec["terms"]) if p.objective_spec else False
        requested |= any(source_definition(name)["kind"] in frontier_features for name in p.provenance.get("source_feature_bounds", {}))
        certificate = r["cp_lowering_frontier"]
        if requested:
            expected=[{key:row[key] for key in CP_CERTIFICATE_KEYS} for row in trace]
            require(isinstance(certificate,dict) and certificate=={"kind":"exact-fourk-mask-ready-cutoff-last-commit-overlap",
                "slab_bytes":SLAB_BYTES,"arena_slabs":p.arena_slabs,"events":expected},"Exact V5 readiness/read-mask certificate changed")
        else:
            require(certificate is None, "Unrequested lowering CP certificate")
    require(all(minimum <= r["source_features"][kind] <= maximum
                for kind, (minimum, maximum) in p.provenance.get("source_feature_bounds", {}).items()),
            "Selected source order is outside frozen feature bounds")
    expected_source_terms = {json.dumps(t["source"], sort_keys=True): _source_value(p, r, t["source"])
                             for t in p.objective_spec["terms"] if "source" in t} if p.objective_spec else {}
    require(r.get("cp_source_features", {}) == expected_source_terms, "CP source feature certificate changed")
    require(r["objective"] == evaluate_objective(p, ordered, r) and r["objective_features"] == objective_features(p, ordered, r), "Proxy objective/feature differs from selected allocation/order")
    require(r["objective_model"] == {"kind": "affine_features" if p.objective_spec else "action_coefficient_sum",
                                    "spec": p.objective_spec, "unit": p.provenance["objective_unit"]}, "Objective model record changed")
    for before, after in p.precedence_edges:
        if before in events and after in events:
            require(events[before]["end"] <= events[after]["start"], "Precedence/state-chain order violated")
    for e in ordered:
        if e["kind"] == "recover_p":
            require(any(pv["kind"] in ("pv_first", "pv_next") and pv["tile"] == e["tile"] and pv["end"] <= e["start"]
                        for pv in ordered), "P recovery must follow an actual stripe PV use")
    expected_states = [dict(stripe=s, tier="reg", bytes=OUTPUT_STRIPE_BYTES, start=events[chain[0]]["start"],
        end=events[chain[-1]]["end"], first_event=chain[0], last_event=chain[-1], effect_order=list(chain))
        for s, chain in enumerate(p.state_chains)]
    require(r["output_states"] == expected_states, "Output stripe state/lifetime changed")
    expected_initial = {"gmem:"+v.name for v in p.values if v.role in ("q", "k", "v")
        and not (p.provenance["backend_spec"]["full_value_parent"] and v.role=="v" and v.stripe is not None)}
    require({c["id"] for c in copies.values() if c["producer"] is None} == expected_initial, "Initial inputs invented or missing")
    for identity, c in copies.items():
        require(c["value"] in values and c["tier"] in ("gmem", "smem", "reg"), "Unknown value/tier")
        v = values[c["value"]]
        require(isinstance(c["workspace"], bool) and c["workspace"] == (v.role == "workspace"), "Workspace marker changed")
        size = v.logical_bytes if c["tier"] == "gmem" else _occupied(v, c["tier"])
        require((c["role"], c["tile"], c["stripe"], c["layout"], c["logical_bytes"], c["nbytes"], c["occupied_bytes"]) ==
                (v.role, v.tile, v.stripe, v.layout, v.logical_bytes, v.logical_bytes, size), "Copy role/layout/byte contract changed")
        require(all(integer(c[k]) for k in ("start", "available", "end")) and 0 <= c["start"] <= c["available"] <= c["end"] <= finish, "Invalid copy lifetime")
        if c["producer"] is None:
            require(identity == "gmem:"+v.name and c["tier"] == "gmem" and c["start"] == c["available"] == 0 and c["end"] == finish, "Invented global backup")
        elif c["workspace"]:
            require(c["producer"] in events and identity in events[c["producer"]]["workspace_copies"], "Orphan workspace")
        else:
            require(c["producer"] in events and identity in events[c["producer"]]["outputs"], "Orphan produced copy")
            require(uses[identity] and c["end"] == max(c["available"], *uses[identity]), "Unused output or unjustified lifetime")
        if c["tier"] == "smem":
            alignment = (1 if c["workspace"] else p.shared_alignment_slabs_by_role[v.role])*SLAB_BYTES
            require(integer(c["offset_bytes"]) and c["offset_bytes"] % alignment == 0 and c["offset_bytes"] + size <= p.arena_slabs*SLAB_BYTES, "Invalid backend shared alignment/address")
        else:
            require(c["offset_bytes"] is None, "Physical RF/global addresses are not modeled")
    peaks = dict(reg=0, smem=0, open_output_stripes=0)
    points = sorted({0, finish, *(c["start"] for c in copies.values()), *(c["end"] for c in copies.values()),
                     *(e["start"] for e in ordered), *(e["end"] for e in ordered)})
    for point in points:
        live = [c for c in copies.values() if c["start"] <= point < c["end"]]
        states = [s for s in expected_states if s["start"] <= point < s["end"]]
        rf = sum(c["occupied_bytes"] for c in live if c["tier"] == "reg") + sum(s["bytes"] for s in states)
        rf += sum(e["workspace_reg_bytes"] for e in ordered if e["start"] <= point < e["end"])
        require(rf <= p.register_budget_bytes, "RF payload capacity exceeded")
        shared = [c for c in live if c["tier"] == "smem"]
        for i, a in enumerate(shared):
            for b in shared[i+1:]:
                require(max(a["offset_bytes"], b["offset_bytes"]) >= min(a["offset_bytes"]+a["occupied_bytes"], b["offset_bytes"]+b["occupied_bytes"]), "Live shared addresses overlap")
        peaks["reg"] = max(peaks["reg"], rf)
        peaks["smem"] = max(peaks["smem"], sum(c["occupied_bytes"] for c in shared))
        peaks["open_output_stripes"] = max(peaks["open_output_stripes"], len(states))
    counts = {}
    for e in ordered:
        counts[e["name"]] = counts.get(e["name"], 0)+1
    exchange_certificate = {}
    if "residency_exchange_v78" in p.provenance:
        from .exchange import verify
        exchange_certificate["residency_exchange_v78"] = verify(p.provenance["residency_exchange_v78"], r)
    return dict(valid=True, peak_bytes={k: peaks[k] for k in ("reg", "smem")}, peak_open_output_stripes=peaks["open_output_stripes"],
                output_states=expected_states, row_snapshot_bytes={v.name: v.register_bytes for v in p.values if v.role in ("m", "l", "alpha")},
                control_storage_bytes=0, executions=counts,
                recomputations={k: n-1 for k, n in counts.items() if k.startswith("qk_") and n > 1},
                pure_p_recoveries=sum(n for k, n in counts.items() if k.startswith("recover_p_")),
                q_global_loads=counts.get("copy_Q", 0), source_order_steps=finish,
                objective=r["objective"], objective_unit=r["objective_unit"], objective_features=r["objective_features"],
                note="Ordinal source dependency/address/state replay with a separate scalar/affine objective; not compiled completion, physical RF release or measured latency",
                **exchange_certificate)
