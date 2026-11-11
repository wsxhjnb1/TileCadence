"""CPU regressions for shared residency exchange and CuTe lowering."""

from copy import deepcopy
import re
from types import SimpleNamespace
import unittest

from tilecadence import model
from tilecadence import emit_exchange as emitter


def span(copy):
    start = copy["offset_bytes"]
    return start, start + copy["occupied_bytes"]


class CheckpointFrontier(unittest.TestCase):
    def test_checkpoint_barriers_neither_complete_async_copies_nor_hide_old_reads(self):
        # Exercise the lowering oracle directly with deterministic physical aliases.
        problem = SimpleNamespace(provenance={
            "residency_exchange_v78": {},
            "execution_mode": "source_order",
            "backend_spec": {"close_workspace_smem_bytes": 8192},
        })
        shared = [
            ("issue_q", "q", None, 0, 16384),
            ("issue_v", "v", 0, 16384, 4096),
            ("store_p", "p", None, 0, 4096),
            ("issue_k", "k", None, 0, 8192),
        ]
        copies = [dict(id=name, producer=name, role=role, stripe=stripe, tier="smem",
                       offset_bytes=offset, occupied_bytes=width)
                  for name, role, stripe, offset, width in shared]
        sequence = [
            ("issue_q", "copy_g2s", [], ["issue_q"]),
            ("read_q", "load_s2r", ["issue_q"], ["q_rf"]),
            ("issue_v", "copy_g2s", [], ["issue_v"]),
            ("store_p", "store_r2s", [], ["store_p"]),
            ("read_p", "load_s2r", ["store_p"], ["p_rf"]),
            ("read_v", "load_s2r", ["issue_v"], ["v_rf"]),
            ("read_p_again", "load_s2r", ["store_p"], ["p_rf"]),
            ("read_v_again", "load_s2r", ["issue_v"], ["v_rf"]),
            ("issue_k", "copy_g2s", [], ["issue_k"]),
        ]
        copies.extend({"id": name, "tier": "reg"} for name in ("q_rf", "p_rf", "v_rf"))
        events = [dict(id=name, kind=kind, start=step, end=step+1, inputs=inputs, outputs=outputs)
                  for step, (name, kind, inputs, outputs) in enumerate(sequence)]
        trace = {row["id"]: row for row in model.lowering_trace(problem, {"copies": copies, "events": events})}

        self.assertEqual(trace["store_p"]["shared_read_write_hazards"], [["store_p", "issue_q"]])
        self.assertEqual(trace["store_p"]["read_mask_after"], 0)
        self.assertEqual(trace["read_p"]["read_mask_after"], 1)
        self.assertEqual(trace["read_p"]["issued_groups_after"], 2)
        self.assertEqual(trace["read_p"]["ready_groups_after"], 1)
        self.assertTrue(trace["read_v"]["wait_before_load"])
        self.assertFalse(trace["read_v_again"]["wait_before_load"])
        self.assertEqual(trace["issue_k"]["read_mask_before"], 17)
        self.assertEqual(trace["issue_k"]["shared_read_write_hazards"], [["issue_k", "store_p"]])
        self.assertTrue(trace["issue_k"]["release_before_write"])


class ResidencyExchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tilecadence import exchange as exchange

        cls.problem = exchange.build_problem("relocated_two_p")
        cls.result = model.solve(cls.problem, max_time_seconds=30)
        if cls.result["status"] not in ("FEASIBLE", "OPTIMAL"):
            raise AssertionError(cls.result["status"])
        cls.copies = {copy["id"]: copy for copy in cls.result["copies"]}
        cls.events = {event["id"]: event for event in cls.result["events"]}
        cls.trace = {row["id"]: row for row in cls.result["lowering_trace"]}

    def test_disjoint_probability_checkpoints_reuse_the_live_donors_range(self):
        donor = self.copies["copy_Q#0:out0"]
        restored = self.copies["copy_Q#1:out0"]
        p0 = self.copies["store_P0#0:out0"]
        p1 = self.copies["store_P1#0:out0"]
        donor_lo, donor_hi = span(donor)
        restore_lo, restore_hi = span(restored)
        self.assertTrue(restore_hi <= donor_lo or donor_hi <= restore_lo)
        self.assertLessEqual(donor["end"], p0["start"])
        self.assertLessEqual(p0["available"], restored["start"])
        self.assertLessEqual(restored["end"], p1["start"])
        self.assertLess(p1["available"], min(p0["end"], p1["end"]))
        p_spans = sorted([span(p0), span(p1)])
        self.assertLessEqual(p_spans[0][1], p_spans[1][0])
        for lo, hi in p_spans:
            self.assertLessEqual(donor_lo, lo)
            self.assertLessEqual(hi, donor_hi)
            self.assertEqual(hi - lo, 4096)

        self.assertEqual(self.events["load_Q#0"]["inputs"], [donor["id"]])
        self.assertEqual(self.events["load_Q#1"]["inputs"], [restored["id"]])
        self.assertIn("load_Q#0:out0", self.events["qk_0#0"]["inputs"])
        self.assertIn("load_Q#1:out0", self.events["qk_1#0"]["inputs"])
        for tile, checkpoint in enumerate((p0, p1)):
            self.assertEqual(self.events[f"load_P{tile}#0"]["inputs"], [checkpoint["id"]])
            self.assertIn(f"load_P{tile}#0:out0", self.events[f"pv_{tile}_1#0"]["inputs"])

    def test_lowering_keeps_checkpoint_barriers_separate_from_copy_acquire(self):
        for tile in (0, 1):
            store = self.trace[f"store_P{tile}#0"]
            load = self.trace[f"load_P{tile}#0"]
            checkpoint = self.copies[f"store_P{tile}#0:out0"]
            bit = 1 << (checkpoint["offset_bytes"] // 4096)
            for row in (store, load):
                self.assertEqual(row["ready_cutoff_before"], row["ready_cutoff_after"])
                self.assertEqual(row["ready_groups_before"], row["ready_groups_after"])
                self.assertEqual(row["issued_groups_before"], row["issued_groups_after"])
                self.assertFalse(row["wait_before_load"])
            self.assertEqual(store["read_mask_after"], 0)
            self.assertEqual(load["read_mask_after"], load["read_mask_before"] | bit)

        source, manifest = emitter.source_for(self.problem, self.result)
        self.assertEqual(source.count("issue_g2s<Element,64,128>"), 2)
        self.assertEqual(source.count("store_checkpoint("), 2)
        self.assertEqual(source.count("load_checkpoint<Element>("), 2)
        self.assertEqual(manifest["source_ready_mask_certificate"], self.result["lowering_trace"])
        lines = source.splitlines()
        for tile in (0, 1):
            event = f"load_P{tile}#0"
            bounds = manifest["event_source_lines"][event]
            body = "\n".join(lines[bounds["comment_line"]-1:bounds["last_line"]])
            match = re.search(r"load_checkpoint<Element>\((s\d+),mma\)", body)
            self.assertIsNotNone(match)
            address = self.copies[f"store_P{tile}#0:out0"]["offset_bytes"]
            declaration = f"auto {match[1]}=make_tensor(make_smem_ptr(reinterpret_cast<Element*>(arena+{address}))"
            self.assertIn(declaration, source)

    def test_replay_rejects_missing_checkpoint_read(self):
        tampered = deepcopy(self.result)
        load = next(row for row in tampered["lowering_trace"] if row["id"] == "load_P0#0")
        checkpoint = self.copies["store_P0#0:out0"]
        load["read_mask_after"] &= ~(1 << (checkpoint["offset_bytes"] // 4096))
        with self.assertRaises(ValueError):
            model.replay(self.problem, tampered)


if __name__ == "__main__":
    unittest.main()
