"""Generate and verify controlled shared-memory residency exchange plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import model as v52


ARMS = ("retain", "one", "multi", "relocated_two_p")
PROVENANCE_KEY = "residency_exchange_v78"
INITIAL_Q = "copy_Q#0:out0"
RESTORED_Q = "copy_Q#1:out0"
P0 = "store_P0#0:out0"
P1 = "store_P1#0:out0"
V = "copy_V0_1#0:out0"
RECIPIENTS = {"retain": (), "one": (P0,), "multi": (P0, V), "relocated_two_p": (P0, P1)}


def build_contract(arm):
    v52.require(arm in ARMS, f"Unknown residency arm: {arm}")
    return {
        "revision": "v78",
        "arm": arm,
        "arena_slabs": 8,
        "slab_bytes": v52.SLAB_BYTES,
        "donor": INITIAL_Q,
        "restore": RESTORED_Q if arm != "retain" else None,
        "recipients": list(RECIPIENTS[arm]),
        "scope": "M64/N64/D128 split-V, shared128 output, exact two QK, no P rematerialization",
    }


def _arm(contract):
    arm = contract.get("arm")
    v52.require(contract == build_contract(arm), "Residency contract differs from the frozen arm")
    return arm


def build_problem(arm):
    contract = build_contract(arm)
    # Source-level proxies only. Calibrated comparisons require compiled GPU timings.
    objective = {
        "unit": "cost_ticks",
        "constant": 0,
        "terms": [
            {"feature": "source_actions", "coefficient": 128, "match": {}},
            {"feature": "rf_payload_kib_steps", "coefficient": 1, "source": {"kind": "rf_kib_steps"}},
        ],
        "provenance": {"scope": "Same uncalibrated action-count and RF liveness proxy for every arm"},
    }
    return v52.build_problem(
        arena_slabs=contract["arena_slabs"],
        allow_score_spill=False,
        allow_probability_spill=True,
        max_qk_instances=1,
        max_p_recoveries=0,
        objective_spec=objective,
        full_value_parent=False,
        close_workspace_smem_bytes=8192,
        provenance={PROVENANCE_KEY: contract},
    )


def constrain(model, contract, actions, copies, binds):
    """Constrain V52 CP variables without changing its common tile graph."""
    arm = _arm(contract)
    by_copy = {copy["id"]: copy for copy in copies}
    by_bind = {(row["consumer"], row["port"], row["copy"]): row["use"] for row in binds}

    def bind(consumer, port, source):
        model.add(by_bind[consumer, port, source] == 1)

    def before(first, second):
        model.add(actions[first]["end"] <= actions[second]["start"])

    allowed_repeats = {"load_Q#1"}
    if arm != "retain":
        allowed_repeats.add("copy_Q#1")
    for identity, variables in actions.items():
        if not identity.endswith("#0") and identity not in allowed_repeats:
            model.add(variables["use"] == 0)

    checkpoint_tiles = {0, 1} if arm == "relocated_two_p" else ({0} if arm != "retain" else set())
    for tile in (0, 1):
        if tile not in checkpoint_tiles:
            model.add(actions[f"store_P{tile}#0"]["use"] == 0)
            model.add(actions[f"load_P{tile}#0"]["use"] == 0)

    bind("load_Q#0", 0, INITIAL_Q)
    bind("qk_0#0", 0, "load_Q#0:out0")
    if arm == "retain":
        # The solver may reuse the first RF Q or load again from the sole shared Q.
        return

    bind("load_Q#1", 0, RESTORED_Q)
    bind("qk_1#0", 0, "load_Q#1:out0")
    before("qk_0#0", "copy_Q#1")
    donor = by_copy[INITIAL_Q]
    restored = by_copy[RESTORED_Q]
    width = donor["occupied_bytes"] // v52.SLAB_BYTES
    restore_above = model.new_bool_var("v78:restore_above_donor")
    model.add(restored["offset"] >= donor["offset"] + width).only_enforce_if(restore_above)
    model.add(donor["offset"] >= restored["offset"] + width).only_enforce_if(restore_above.Not())

    def checkpoint(tile, source_port):
        store = f"store_P{tile}#0"
        load = f"load_P{tile}#0"
        stats_p = f"stats_{tile}#0:out{source_port}"
        bind(store, 0, stats_p)
        bind(load, 0, store + ":out0")
        bind(f"pv_{tile}_0#0", 0, stats_p)
        bind(f"pv_{tile}_1#0", 0, load + ":out0")
        before(f"pv_{tile}_0#0", load)

    checkpoint(0, 2)
    before("pv_0_0#0", "qk_1#0")
    before("qk_1#0", "load_P0#0")
    p0 = by_copy[P0]
    model.add(p0["available"] < restored["start"])

    if arm == "multi":
        v = by_copy[V]
        bind("load_V0_1#0", 0, V)
        bind("pv_0_1#0", 1, "load_V0_1#0:out0")
        model.add(v["available"] < restored["start"])
        before("copy_Q#1", "load_V0_1#0")
    elif arm == "relocated_two_p":
        checkpoint(1, 3)
        before("store_P1#0", "load_P0#0")

    recipients = RECIPIENTS[arm]
    for identity in recipients:
        copy = by_copy[identity]
        slots = copy["occupied_bytes"] // v52.SLAB_BYTES
        model.add(donor["end"] <= copy["start"])
        model.add(donor["offset"] <= copy["offset"])
        model.add(copy["offset"] + slots <= donor["offset"] + width)

    # No undeclared occupants of donor addresses during the pre-restore window.
    window_start = p0["available"]
    if arm == "multi":
        window_start = model.new_int_var(0, len(actions), "v78:joint_recipient_start")
        model.add_max_equality(window_start, [p0["available"], by_copy[V]["available"]])
    for copy in copies:
        identity = copy["id"]
        if copy["ref"][1] != "smem" or identity in (INITIAL_Q, RESTORED_Q, *recipients):
            continue
        slots = copy["occupied_bytes"] // v52.SLAB_BYTES
        alternatives = (
            copy["end"] <= window_start,
            copy["start"] >= restored["start"],
            copy["offset"] + slots <= donor["offset"],
            copy["offset"] >= donor["offset"] + width,
        )
        choices = []
        for i, relation in enumerate(alternatives):
            choice = model.new_bool_var(f"v78:outside_window:{identity}:{i}")
            model.add(relation).only_enforce_if(choice)
            choices.append(choice)
        model.add_bool_or(choices).only_enforce_if(copy["use"])


def verify(contract, result):
    """Independently validate actual selected bindings, intervals, and byte ranges."""
    arm = _arm(contract)
    v52.require(result.get("status") in ("OPTIMAL", "FEASIBLE"), "Residency verification needs a solved plan")
    events = {row["id"]: row for row in result["events"]}
    copies = {row["id"]: row for row in result["copies"]}

    def need(condition, message):
        v52.require(condition, f"{arm}: {message}")

    def span(copy):
        lo = copy["offset_bytes"]
        nbytes = copy["occupied_bytes"]
        need(copy["tier"] == "smem" and v52.integer(lo) and v52.integer(nbytes, 1)
             and lo % v52.SLAB_BYTES == 0 and nbytes % v52.SLAB_BYTES == 0
             and lo + nbytes <= contract["arena_slabs"] * v52.SLAB_BYTES,
             f"invalid arena span for {copy['id']}")
        return [lo, lo + nbytes]

    def check_bind(event, port, source):
        need(event in events and events[event]["inputs"][port] == source and source in copies,
             f"{event} must consume {source} at port {port}")

    need({e["id"] for e in events.values() if e["kind"] == "qk"} == {"qk_0#0", "qk_1#0"},
         "exactly one required QK per tile")
    need(not any(e["kind"] == "recover_p" for e in events.values()), "P recomputation is excluded")
    permitted_repeats = {"load_Q#1"} if arm == "retain" else {"load_Q#1", "copy_Q#1"}
    need(all(identity.endswith("#0") or identity in permitted_repeats for identity in events),
         "unexpected repeated action outside the common graph")
    q_copies = [c["id"] for c in copies.values() if c["tier"] == "smem" and c["role"] == "q"]
    expected_q = {INITIAL_Q} if arm == "retain" else {INITIAL_Q, RESTORED_Q}
    need(set(q_copies) == expected_q, "unexpected number or identity of shared Q copies")
    check_bind("load_Q#0", 0, INITIAL_Q)
    check_bind("qk_0#0", 0, "load_Q#0:out0")
    donor = copies[INITIAL_Q]
    donor_span = span(donor)
    need(donor["occupied_bytes"] == 16384, "full Q donor must occupy 16 KiB")
    certificate = {
        "contract": contract,
        "donor_span_bytes": donor_span,
        "original_q_shared_source_steps": [donor["start"], donor["end"]],
        "recipients": [],
        "scope": "Address reservation and source-order dependencies; async readiness, compiled work and speed require device evidence",
    }

    selected_checkpoints = {e["id"] + ":out0" for e in events.values() if e["kind"] == "store_r2s"}
    need(selected_checkpoints == {p for p in RECIPIENTS[arm] if p.startswith("store_")},
         "unexpected checkpoint path")
    if arm == "retain":
        second_source = events["qk_1#0"]["inputs"][0]
        if second_source != "load_Q#0:out0":
            check_bind("qk_1#0", 0, "load_Q#1:out0")
            check_bind("load_Q#1", 0, INITIAL_Q)
        certificate["qk1_q_register_source"] = second_source
        return certificate

    check_bind("load_Q#1", 0, RESTORED_Q)
    check_bind("qk_1#0", 0, "load_Q#1:out0")
    restored = copies[RESTORED_Q]
    restore_span = span(restored)
    need(restored["occupied_bytes"] == donor["occupied_bytes"] and
         (restore_span[1] <= donor_span[0] or donor_span[1] <= restore_span[0]),
         "restore Q must use a disjoint address range")
    need(events["qk_0#0"]["end"] <= restored["start"], "restore must follow the original QK")
    need(copies["load_Q#0:out0"]["end"] == events["qk_0#0"]["end"], "original Q RF held beyond first QK")
    certificate["restore_span_bytes"] = restore_span
    certificate["restore_issue_source_step"] = restored["start"]

    def check_checkpoint(tile, source_port):
        store = f"store_P{tile}#0"
        load = f"load_P{tile}#0"
        stats_p = f"stats_{tile}#0:out{source_port}"
        check_bind(store, 0, stats_p)
        check_bind(load, 0, store + ":out0")
        check_bind(f"pv_{tile}_0#0", 0, stats_p)
        check_bind(f"pv_{tile}_1#0", 0, load + ":out0")
        need(events[f"pv_{tile}_0#0"]["end"] <= events[load]["start"], "checkpoint read precedes original P use")

    check_checkpoint(0, 2)
    need(events["pv_0_0#0"]["end"] <= events["qk_1#0"]["start"] <= events["qk_1#0"]["end"]
         <= events["load_P0#0"]["start"], "P0 checkpoint must bridge the second QK")
    if arm == "multi":
        check_bind("load_V0_1#0", 0, V)
        check_bind("pv_0_1#0", 1, "load_V0_1#0:out0")
        need(events["copy_Q#1"]["end"] <= events["load_V0_1#0"]["start"], "V must remain reserved through Q restore issue")
    elif arm == "relocated_two_p":
        check_checkpoint(1, 3)
        need(events["store_P1#0"]["end"] <= events["load_P0#0"]["start"], "P0 must remain live across P1 store")

    recipients = [copies[identity] for identity in RECIPIENTS[arm]]
    for copy in recipients:
        lo, hi = span(copy)
        need(donor_span[0] <= lo < hi <= donor_span[1] and hi - lo == v52.SLAB_BYTES,
             f"recipient {copy['id']} must occupy one slab in the vacated donor")
        need(donor["end"] <= copy["start"] and copy["available"] < copy["end"],
             f"recipient {copy['id']} has invalid donor reuse or no later consumer")
        certificate["recipients"].append({
            "copy": copy["id"],
            "span_bytes": [lo, hi],
            "source_steps": [copy["start"], copy["end"]],
            "producer_end": copy["available"],
        })

    active_before = recipients[:1] if arm == "relocated_two_p" else recipients
    pre_start = max(copy["available"] for copy in active_before)
    pre_end = restored["start"]
    need(pre_start < pre_end <= min(copy["end"] for copy in active_before), "no pre-restore recipient window")
    certificate["pre_restore_reservation_source_steps"] = [pre_start, pre_end]
    for copy in copies.values():
        if copy["tier"] != "smem" or copy["id"] in (INITIAL_Q, RESTORED_Q, *RECIPIENTS[arm]):
            continue
        lo, hi = span(copy)
        need(copy["end"] <= pre_start or pre_end <= copy["start"] or hi <= donor_span[0] or donor_span[1] <= lo,
             f"undeclared occupant {copy['id']} in the pre-restore window")

    joint_start = max(copy["available"] for copy in recipients)
    joint_end = min(copy["end"] for copy in recipients)
    need(joint_start < joint_end, "no common recipient reservation window")
    if len(recipients) == 2:
        a, b = (span(copy) for copy in recipients)
        need(a[1] <= b[0] or b[1] <= a[0], "simultaneous recipients overlap in address")
    if arm == "relocated_two_p":
        need(restored["available"] <= joint_start, "P1 is not available before Q is restored elsewhere")
    certificate["joint_reservation_source_steps"] = [joint_start, joint_end]
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--no-emit", action="store_true")
    args = parser.parse_args()
    v52.require(0 < args.seconds <= 120, "Per-arm CP-SAT budget must be between zero and 120 seconds")
    arms = args.arm or ARMS
    v52.require(len(set(arms)) == len(arms), "Each arm may only be generated once")
    args.out.mkdir(parents=True, exist_ok=False)
    failures = []
    for arm in arms:
        role = args.out / arm
        role.mkdir()
        problem = build_problem(arm)
        try:
            result = v52.solve(problem, max_time_seconds=args.seconds, artifact_dir=role / "solver")
            certificate = verify(problem.provenance[PROVENANCE_KEY], result)
            (role / "residency_certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
            if not args.no_emit:
                from .emit_exchange import emit
                emit(problem, result, role / "emitted", opaque_recovery=False)
            print(f"{arm}: {result['status']}; {result['finish_step']} source steps", flush=True)
        except (ValueError, RuntimeError) as error:
            (role / "failure.txt").write_text(str(error) + "\n")
            failures.append(arm)
            print(f"{arm}: {error}", flush=True)
    if failures:
        raise SystemExit("Unsuccessful arms: " + ", ".join(failures))


if __name__ == "__main__":
    main()
