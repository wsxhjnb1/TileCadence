"""CPU-side plans and selection for the long-context Qwen3 A+B study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


CASE_DIMENSIONS = {f"model_b{batch}_n{length}": (batch, length)
                   for batch in (1, 4) for length in (8192, 32768)}
CASES = tuple(CASE_DIMENSIONS)
PHASES = ("prefill", "decode")
N_CANDIDATES = 4
N_PAIRS = 3
BASELINE = "pinned upstream native DEFAULT FA2"
CAPTURE_INDEX_SHA256 = "b2232e72df41087622da76b8f52a7b797b06f0b0f9ab76713dd774f71c71cce7"
BASE_MANIFEST_SHA256 = "46bf76e3d4748dad6240cfe74c01b80f5f5b94d99ea8bacfcb2591ff47379e9f"
VENDOR_MANIFEST_SHA256 = "919ac9c0333363d56fede38963da44266ea5d996f9ec178da4c7c55f02141e45"
CATALOG_SCHEMA = "qwen3-long-ab-catalog-v14"
PLAN_SCHEMA = "qwen3-long-ab-dev-plan-v14"
EVAL_SCHEMA = "qwen3-long-ab-dev-evaluation-v14"
SELECTION_SCHEMA = "qwen3-long-ab-dev-selection-v14"
RESOURCE_FEATURES = (
    "main_registers_per_thread",
    "main_dynamic_shared_bytes",
    "main_static_shared_bytes",
    "main_local_bytes_per_thread",
    "main_theoretical_active_ctas_per_sm",
)
VALID_STATES = ("passed", "compile_failed", "numerical_failed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_pins():
    here = Path(__file__).resolve().parent
    return {name: sha(here / name) for name in (
        "qwen3_long_ab_controller_v14.py", "qwen3_long_ab_dev_v14.py", "runtime.py",
    )}


def validate_catalog(catalog):
    require(catalog.get("schema") == CATALOG_SCHEMA, "Expected the frozen B-domain catalog")
    require(catalog.get("baseline") == BASELINE, "Only actual native DEFAULT FA2 is the baseline")
    pins = catalog["b_source_pins"]
    require(isinstance(pins, dict) and pins and all(is_sha(value) for value in pins.values()),
            "B generator/source identity must be pinned")
    candidates = catalog["candidates"]
    ids = [row["candidate_id"] for row in candidates]
    require(len(ids) == len(set(ids)) and all(re.fullmatch(r"[a-z][a-z0-9_]{0,55}", x) for x in ids),
            "Each candidate requires a unique bounded identifier")
    require(all(row["phase"] in PHASES for row in candidates), "Unexpected phase")
    for phase in PHASES:
        rows = [row for row in candidates if row["phase"] == phase]
        require(len(rows) >= 2 * N_CANDIDATES, f"Need eight distinct B-domain options for {phase}")
        keys = set(rows[0]["features"])
        require(keys, "A requires numeric features describing B execution choices")
        for row in rows:
            require(set(row["features"]) == keys and all(is_number(x) for x in row["features"].values()),
                    f"Inconsistent source features for {row['candidate_id']}")
            require(isinstance(row["parameters"], dict) and row["parameters"], "A proposal must bind its B inputs")
            controls = row["controls"]
            require(set(controls) == {"mapping", "num_splits"} and type(controls["mapping"]) is int
                    and controls["mapping"] > 0 and type(controls["num_splits"]) is int
                    and 0 <= controls["num_splits"] <= 128, "Expected bounded candidate runtime controls")
        fingerprints = {json.dumps([r["parameters"], r["controls"]], sort_keys=True) for r in rows}
        require(len(fingerprints) == len(rows), "Duplicate B requests would consume a search slot")
    return {row["candidate_id"]: row for row in candidates}


def source_distances(candidates):
    names = sorted(candidates[0]["features"])
    vectors = {row["candidate_id"]: [row["features"][name] for name in names] for row in candidates}
    ranges = [max(column) - min(column) or 1 for column in zip(*vectors.values())]

    def distance(left, right):
        differences = ((x - y) / scale for x, y, scale in zip(vectors[left], vectors[right], ranges))
        return math.sqrt(statistics.mean(value ** 2 for value in differences))

    return distance


def choose_seed(candidates):
    ids = sorted(row["candidate_id"] for row in candidates)
    distance = source_distances(candidates)
    selected = [min(ids, key=lambda x: (statistics.mean(distance(x, y) for y in ids), x))]
    while len(selected) < N_CANDIDATES:
        available = [x for x in ids if x not in selected]
        selected.append(min(available, key=lambda x: (-min(distance(x, y) for y in selected), x)))
    return selected


def make_plan(catalog_path, round_id, selected, *, previous=None, evaluation=None, fit=None):
    return dict(schema=PLAN_SCHEMA, round_id=round_id, baseline=BASELINE,
                use="development_only", cases=list(CASES), pairs=N_PAIRS,
                capture_index_sha256=CAPTURE_INDEX_SHA256, base_manifest_sha256=BASE_MANIFEST_SHA256,
                source_pins=source_pins(), catalog_sha256=sha(catalog_path),
                parent_plan_sha256=sha(previous) if previous else None,
                parent_evaluation_sha256=sha(evaluation) if evaluation else None,
                selected=selected, feedback=fit)


def validate_plan(plan, catalog=None):
    require(plan["schema"] == PLAN_SCHEMA and type(plan["round_id"]) is int and plan["round_id"] in (0, 1),
            "Expected one of the two frozen DEV rounds")
    require(plan["baseline"] == BASELINE and plan["use"] == "development_only", "DEV comparison changed")
    require(plan["cases"] == list(CASES) and plan["pairs"] == N_PAIRS, "Frozen case/replicate budget differs")
    require(plan["capture_index_sha256"] == CAPTURE_INDEX_SHA256 and plan["base_manifest_sha256"] == BASE_MANIFEST_SHA256,
            "Frozen real input/native library identity differs")
    require(plan["source_pins"] == source_pins(), "DEV/controller sources changed after freeze")
    require(is_sha(plan["catalog_sha256"]), "Missing B-domain identity")
    prior = (plan["parent_plan_sha256"], plan["parent_evaluation_sha256"])
    require(all(is_sha(value) for value in prior) if plan["round_id"] else prior == (None, None),
            "Round 1 must bind round 0 DEV feedback")
    selected = plan["selected"]
    require(len({row["candidate_id"] for row in selected}) == len(selected) == len(PHASES) * N_CANDIDATES,
            "Wrong distinct candidate budget")
    for phase in PHASES:
        require(sum(row["phase"] == phase for row in selected) == N_CANDIDATES, "Wrong phase budget")
    if catalog is not None:
        require(all(catalog.get(row["candidate_id"]) == row for row in selected),
                "A-selected B parameters differ from the frozen catalog")
    return selected


def role_order(round_id, trial, case, pair):
    baseline_first = (round_id + trial + CASES.index(case) + pair) % 2 == 0
    return ["official_default", "candidate"] if baseline_first else ["candidate", "official_default"]


def log_cost(row):
    return statistics.mean(math.log(sample["candidate_ns"] / sample["native_ns"]) for sample in row["pairs"])


def validate_evaluation(evaluation, plan, plan_path):
    require(evaluation["schema"] == EVAL_SCHEMA and evaluation["state"] == "complete",
            "Retained complete DEV evaluation required")
    require(evaluation["plan_sha256"] == sha(plan_path) and evaluation["round_id"] == plan["round_id"],
            "DEV results belong to a different exact plan")
    require(evaluation["baseline"] == BASELINE and evaluation["use"] == "development_only"
            and evaluation["measurement"] == "cuda_event_single_attention_call_including_combine",
            "DEV labels require matched complete-call measurements")
    require(evaluation["capture_index_sha256"] == CAPTURE_INDEX_SHA256 and is_sha(evaluation["build_batch_sha256"])
            and is_sha(evaluation["raw_rows_sha256"]),
            "Missing actual DEV capture and executable provenance")
    selected = {row["candidate_id"]: (i, row) for i, row in enumerate(plan["selected"])}
    all_keys = [(row["candidate_id"], row["case"]) for row in evaluation["rows"]]
    require(len(all_keys) == len(set(all_keys)) == len(selected) * len(CASES)
            and set(all_keys) == {(i, case) for i in selected for case in CASES}, "Retain every DEV cell")
    identities = {}
    eligible = []
    for row in evaluation["rows"]:
        identity = row["candidate_id"]
        trial, candidate = selected[identity]
        require(row["phase"] == candidate["phase"] and row["controls"] == candidate["controls"],
                "DEV did not execute the frozen phase/runtime controls")
        require(row["state"] in VALID_STATES and is_sha(row["b_plan_sha256"]),
                "B proof and explicit outcome required for every DEV cell")
        if row["state"] != "passed":
            require(not row.get("pairs"), "Failed DEV cells cannot enter the fit")
            continue
        require(row["full_output_and_lse_checked"] is True and row["checks"]["ok"] is True,
                "Complete same-input numerical gate required before timing")
        require(is_sha(row["module_sha256"]) and is_sha(row["build_receipt_sha256"])
                and row["build_receipt"].startswith("/") and row["source_sha256"]
                and all(is_sha(digest) for digest in row["source_sha256"].values()),
                "Actual compiled identity and Spark receipt path required")
        compiled = (row["module_sha256"], row["build_receipt_sha256"], row["source_sha256"])
        require(compiled == identities.setdefault(identity, compiled), "A trial changed executable across cases")
        resources = row["resources"]
        require(all(type(resources[name]) is int and resources[name] >= 0 for name in RESOURCE_FEATURES)
                and resources["main_binary_version"] == 121
                and resources["main_theoretical_active_ctas_per_sm"] > 0,
                "Complete measured SM121 register/shared/local/CTA resources required")
        require(is_sha(row["capture_sha256"]) and len(row["pairs"]) == N_PAIRS, "Paired input/measurement count differs")
        for pair, sample in enumerate(row["pairs"]):
            require(sample["pair"] == pair and sample["role_order"] == role_order(plan["round_id"], trial, row["case"], pair),
                    "DEV pair order differs from the frozen balanced order")
            require(all(is_number(sample[name]) and sample[name] > 0 for name in ("native_ns", "candidate_ns")),
                    "Finite positive complete-device intervals required")
        eligible.append(row)
    return eligible


def fit_case(candidates, rows, available):
    """Use regularized exploratory prediction with interpolated compiled resources."""
    import numpy as np

    lookup = {row["candidate_id"]: row for row in candidates}
    source_names = sorted(candidates[0]["features"])
    distance = source_distances(candidates)
    observed_ids = [row["candidate_id"] for row in rows]
    actual = {row["candidate_id"]: [row["resources"][k] for k in RESOURCE_FEATURES] for row in rows}

    def source(id_):
        return [lookup[id_]["features"][name] for name in source_names]

    raw = np.asarray([source(id_) + actual[id_] for id_ in observed_ids], dtype=np.float64)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0] = 1
    x = np.column_stack((np.ones(len(rows)), (raw - mean) / scale))
    y = np.asarray([log_cost(row) for row in rows])
    ridge = 10.0
    penalty = np.diag([0.0] + [ridge] * raw.shape[1])
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    predictions = {}
    for id_ in available:
        nearest = sorted(observed_ids, key=lambda old: (distance(id_, old), old))[:2]
        weights = [1 / (distance(id_, old) + 0.125) for old in nearest]
        resources = np.average([actual[old] for old in nearest], axis=0, weights=weights)
        features = np.asarray(source(id_) + resources.tolist())
        estimate = float(beta[0] + ((features - mean) / scale) @ beta[1:])
        predictions[id_] = dict(predicted_log_cost=estimate, nearest_ids=nearest,
                                 nearest_distance=distance(id_, nearest[0]),
                                 forecast_compiled_resources=dict(zip(RESOURCE_FEATURES, resources.tolist())))
    return dict(rows=observed_ids, features=source_names + list(RESOURCE_FEATURES), ridge=ridge,
                means=mean.tolist(), scales=scale.tolist(), coefficients=beta.tolist(), predictions=predictions)


def choose_feedback(candidates, eligible, used):
    available = sorted(row["candidate_id"] for row in candidates if row["candidate_id"] not in used)
    phase = candidates[0]["phase"]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if phase == "decode":
        from ab_long_decode import solve_plan
    else:
        from long_prefill_b_v14 import propose_plan
    selected = []
    fit = {}
    for case in CASES:
        rows = [row for row in eligible if row["case"] == case]
        require(rows, f"No passing hardware feedback for {candidates[0]['phase']} {case}")
        record = fit_case(candidates, rows, available)
        predictions = record["predictions"]
        # One proposal per case. A small, frozen novelty bonus diversifies equal
        # predictions without treating this tiny fit as a calibrated interval.
        scores = {id_: round(1e6 * (predictions[id_]["predicted_log_cost"]
                                   - 0.025 * predictions[id_]["nearest_distance"]))
                  for id_ in available if id_ not in selected}
        if phase == "decode":
            solved = solve_plan(*CASE_DIMENSIONS[case], objective=scores)
            choice = solved["candidate_id"]
        else:
            solved = propose_plan({id_: cost / 1e6 for id_, cost in scores.items()})
            choice = solved["id"]
        require(choice in scores, "B chose a proposal outside the scored unseen domain")
        selected.append(choice)
        record["proposal"] = choice
        record["integer_proposal_costs"] = scores
        record["proposal_solver"] = solved
        fit[case] = record
    return selected, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "feedback", "select"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--catalog", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
    subparsers.choices["feedback"].add_argument("--plan", type=Path, required=True)
    subparsers.choices["feedback"].add_argument("--evaluation", type=Path, required=True)
    subparsers.choices["select"].add_argument("--plans", type=Path, nargs=2, required=True)
    subparsers.choices["select"].add_argument("--evaluations", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    lookup = validate_catalog(read(args.catalog))
    phases = {p: [row for row in lookup.values() if row["phase"] == p] for p in PHASES}
    if args.action == "plan":
        selected = [lookup[id_] for phase in PHASES for id_ in choose_seed(phases[phase])]
        result = make_plan(args.catalog, 0, selected)
    elif args.action == "feedback":
        prior = read(args.plan)
        trials = validate_plan(prior, lookup)
        require(prior["round_id"] == 0 and prior["catalog_sha256"] == sha(args.catalog), "Wrong round-zero plan/catalog")
        rows = validate_evaluation(read(args.evaluation), prior, args.plan)
        used = {row["candidate_id"] for row in trials}
        selected, fit = [], {}
        for phase in PHASES:
            ids, fit[phase] = choose_feedback(phases[phase], [r for r in rows if r["phase"] == phase], used)
            selected.extend(lookup[id_] for id_ in ids)
        result = make_plan(args.catalog, 1, selected, previous=args.plan, evaluation=args.evaluation, fit=fit)
    else:
        eligible, used, evaluations = [], set(), []
        for round_id, (plan_path, evaluation_path) in enumerate(zip(args.plans, args.evaluations)):
            plan = read(plan_path)
            trials = validate_plan(plan, lookup)
            require(plan["round_id"] == round_id and plan["catalog_sha256"] == sha(args.catalog),
                    "Both ordered DEV rounds must use the frozen catalog")
            ids = {row["candidate_id"] for row in trials}
            require(not ids & used, "Feedback round must try new B plans")
            used.update(ids)
            if round_id:
                require(plan["parent_plan_sha256"] == sha(args.plans[0])
                        and plan["parent_evaluation_sha256"] == sha(args.evaluations[0]), "Round 1 feedback ancestry differs")
            evaluation = read(evaluation_path)
            evaluations.append(evaluation)
            eligible.extend(dict(row, development_round=round_id) for row in
                            validate_evaluation(evaluation, plan, plan_path))
        winners, ranks = [], []
        for phase in PHASES:
            for case in CASES:
                rows = [r for r in eligible if r["phase"] == phase and r["case"] == case]
                require(rows, f"No correct compiled B candidate for {phase} {case}")
                rows.sort(key=lambda row: (log_cost(row), row["candidate_id"]))
                winner = rows[0]
                winners.append({**winner, "dev_speedup": math.exp(-log_cost(winner))})
                ranks.append(dict(phase=phase, case=case, ordered_candidate_ids=[r["candidate_id"] for r in rows]))
        result = dict(schema=SELECTION_SCHEMA, state="selected_before_fresh_confirmation", baseline=BASELINE,
                      winners=winners, ranks=ranks, rounds=[0, 1], catalog_sha256=sha(args.catalog),
                      plan_sha256=[sha(p) for p in args.plans],
                      evaluation_sha256=[sha(p) for p in args.evaluations],
                      raw_rows_sha256=[e["raw_rows_sha256"] for e in evaluations],
                      build_batch_sha256=[e["build_batch_sha256"] for e in evaluations],
                      estimator="geometric mean of three paired native/candidate attention intervals",
                      confirmation_used_for_fit=False, confirmation_used_for_selection=False)
    write_new(args.out, result)
    print(json.dumps({"action": args.action, "out": str(args.out), "sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
