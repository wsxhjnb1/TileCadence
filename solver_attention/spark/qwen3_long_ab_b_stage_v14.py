"""Enumerate and emit the actual B source domain on the CPU solver host."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import qwen3_long_ab_controller_v14 as policy


SOLVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLVER_ROOT))

import ab_long_decode as decode
import long_prefill_b_v14 as prefill


PROOF_SCHEMA = "qwen3-long-ab-b-proof-v14"
DECODE_FIELDS = (*decode.SOURCE_FIELDS, "splits")


def source_pins(vendor):
    paths = [Path(__file__).resolve(), Path(decode.__file__).resolve(), Path(prefill.__file__).resolve(),
             SOLVER_ROOT / "ab_long_decode.cu", SOLVER_ROOT / "native_model_v11/prepare.py"]
    paths.extend((SOLVER_ROOT / "native_model_long_prefill_v14").glob("*"))
    for family in ("prefill_natural64", "decode_m16n128"):
        paths.extend((SOLVER_ROOT / "native_model_v11/emitted_v3" / family).glob("*"))
    pins = {path.relative_to(SOLVER_ROOT).as_posix(): policy.sha(path) for path in paths if path.is_file()}
    manifest = vendor / "manifest.json"
    policy.require(policy.sha(manifest) == policy.VENDOR_MANIFEST_SHA256, "Original upstream FA2 manifest required")
    pins["vendor/manifest.json"] = policy.sha(manifest)
    pins["vendor/csrc/flash_attn/src/flash_fwd_kernel.h"] = policy.sha(vendor / "csrc/flash_attn/src/flash_fwd_kernel.h")
    return pins


def enumerate_catalog(vendor):
    pins = source_pins(vendor)
    prefill_rows = prefill.enumerate_plans()
    rows = [dict(candidate_id=row["id"], phase="prefill", parameters={"plan_id": row["id"]},
                 features=row["source_features"], controls=dict(mapping=1, num_splits=0))
            for row in prefill_rows]
    decode_grids = [{row["candidate_id"]: row for row in decode.enumerate_plans(batch, prompt)}
                    for batch, prompt in policy.CASE_DIMENSIONS.values()]
    common = set.intersection(*(set(grid) for grid in decode_grids))
    for id_ in sorted(common):
        row = decode_grids[0][id_]
        parameters = {name: row[name] for name in DECODE_FIELDS}
        policy.require(all({name: grid[id_][name] for name in DECODE_FIELDS} == parameters for grid in decode_grids),
                       "A common decode candidate must represent the same schedule on all cases")
        features = dict(block_n=row["block_n"], warps=row["warps"], retain_q=int(row["q_residency"] != 0),
                        alias_qk=int(row["q_residency"] == 2), cache_always=row["cache_policy"],
                        splits=row["splits"], source_shared_bytes=row["dynamic_shared_bytes"])
        rows.append(dict(candidate_id=id_, phase="decode", parameters=parameters,
                         features=features, controls=dict(mapping=1, num_splits=row["splits"])))
    catalog = dict(schema=policy.CATALOG_SCHEMA, baseline=policy.BASELINE, b_source_pins=pins,
                   cases=list(policy.CASES), candidates=rows, development_labels_used=False)
    policy.validate_catalog(catalog)
    return catalog


def emit_round(args):
    catalog = policy.read(args.catalog)
    plan = policy.read(args.plan)
    trials = policy.validate_plan(plan, policy.validate_catalog(catalog))
    policy.require(plan["catalog_sha256"] == policy.sha(args.catalog), "Exact A catalog required")
    original_pins = catalog["b_source_pins"]
    actual_pins = source_pins(args.vendor)
    stage_key = Path(__file__).resolve().relative_to(SOLVER_ROOT).as_posix()
    amendment = None
    if original_pins != actual_pins:
        prior = args.prior_stage_source
        policy.require(plan["round_id"] == 1 and prior is not None
                       and policy.sha(prior) == original_pins[stage_key]
                       and {**actual_pins, stage_key: original_pins[stage_key]} == original_pins,
                       "Unexpected B solver or emitter source drift")
        amendment = dict(scope="CPU proof JSON round-trip repair", previous_stage_sha256=original_pins[stage_key],
                         actual_stage_sha256=actual_pins[stage_key],
                         repair="Restore prefill precedence tuples before the emitter compares the solved proof",
                         frozen_catalog_and_B_solvers_unchanged=True)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "proofs").mkdir()
    if amendment:
        policy.write_new(args.out / "stage_amendment.json", amendment)
        (args.out / "prior_b_stage.py").write_bytes(args.prior_stage_source.read_bytes())
        (args.out / "actual_b_stage.py").write_bytes(Path(__file__).read_bytes())
    prefill_codes = {id_: 400 + i for i, id_ in enumerate(sorted(
        row["candidate_id"] for row in catalog["candidates"] if row["phase"] == "prefill"))}
    emitted_families = {}
    entries = []
    for trial in trials:
        id_ = trial["candidate_id"]
        phase = trial["phase"]
        parameters = trial["parameters"]
        feedback = None
        if plan["round_id"] == 1:
            records = [item for item in plan["feedback"][phase].values() if item["proposal"] == id_]
            policy.require(len(records) == 1, "Each feedback proposal must retain its fitting case")
            feedback = records[0]["proposal_solver"]
        if phase == "prefill":
            if feedback is not None:
                schedule = feedback["plan"]["schedule"]
                schedule["precedence"] = [tuple(edge) for edge in schedule["precedence"]]
            directory = args.out / "emitted" / id_
            manifest = prefill.emit(directory, parameters["plan_id"], family_code=prefill_codes[id_],
                                    vendor=args.vendor, proposal=feedback)
            policy.require(manifest["source_features"] == trial["features"], "Prefill CP result changed the A source features")
            solved = manifest
        else:
            solved = {case: decode.solve_plan(batch, prompt, fixed=parameters)
                      for case, (batch, prompt) in policy.CASE_DIMENSIONS.items()}
            policy.require(all(row["candidate_id"] == id_ and row["solver"]["status"] == "OPTIMAL"
                               and {name: row[name] for name in DECODE_FIELDS} == parameters for row in solved.values()),
                           "Decode solve differs from the frozen request")
            representative = next(iter(solved.values()))
            family = representative["family"]
            directory = args.out / "emitted" / family
            if family not in emitted_families:
                emitted_families[family] = decode.emit_family(directory, representative, args.vendor)
            manifest = emitted_families[family]
        policy.require(manifest["candidate_mapping"] == trial["controls"]["mapping"], "B emitted a different launch mapping")
        files = {path.name: policy.sha(path) for path in directory.iterdir() if path.is_file()}
        policy.require(all(files[name] == digest for name, digest in manifest["source_sha256"].items()),
                       "Emitted source differs from the solved manifest")
        proof = args.out / "proofs" / f"{id_}.json"
        policy.write_new(proof, dict(schema=PROOF_SCHEMA, state="solved_emitted", candidate_id=id_, phase=phase,
                         parameters=parameters, controls=trial["controls"], solved=solved,
                         feedback_proposal_solver=feedback, source_sha256=files,
                         b_source_pins=actual_pins))
        entries.append(dict(candidate_id=id_, source_dir=directory.relative_to(args.out).as_posix(),
                            source_sha256=files, b_plan=proof.relative_to(args.out).as_posix(),
                            b_plan_sha256=policy.sha(proof)))
    policy.write_new(args.out / "emission.json", dict(schema="qwen3-long-ab-b-emission-v14",
                     state="solved_emitted", round_id=plan["round_id"], plan_sha256=policy.sha(args.plan),
                     catalog_sha256=policy.sha(args.catalog), candidates=entries,
                     stage_amendment_sha256=policy.sha(args.out / "stage_amendment.json") if amendment else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("catalog", "emit"):
        command = subparsers.add_parser(action)
        command.add_argument("--vendor", type=Path, default=SOLVER_ROOT / "vendor_spark_pin_v6/vendor")
        command.add_argument("--out", type=Path, required=True)
    for name in ("catalog", "plan"):
        subparsers.choices["emit"].add_argument(f"--{name}", type=Path, required=True)
    subparsers.choices["emit"].add_argument("--prior-stage-source", type=Path)
    args = parser.parse_args()
    policy.require(not args.out.exists(), "Preserve prior solver outputs")
    if args.action == "catalog":
        policy.write_new(args.out, enumerate_catalog(args.vendor))
    else:
        emit_round(args)


if __name__ == "__main__":
    main()
