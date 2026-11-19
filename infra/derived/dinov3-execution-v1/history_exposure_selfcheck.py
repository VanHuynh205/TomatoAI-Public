"""Identity and transitive-protection checks without model or image execution."""

import csv
import json
import unittest

import history_exposure as audit
from history_exposure import IDENTITY, master, require_identity


class HistoryChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(zip(IDENTITY, ("a.jpg", "a" * 64, "healthy", "s", "field", "g1"), strict=True)),
            dict(zip(IDENTITY, ("b.jpg", "b" * 64, "early_blight", "s", "field", "g2"), strict=True)),
        ]

    def test_reordered_predictions_match(self):
        require_identity(self.rows, list(reversed(self.rows)))

    def test_missing_extra_and_duplicate_predictions_are_rejected(self):
        for actual in (self.rows[:1], self.rows + [self.rows[0]], [self.rows[0]] * 2):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                require_identity(self.rows, actual)
        with self.assertRaises(ValueError):
            require_identity([self.rows[0]] * 2, [self.rows[0]] * 2)

    def test_each_identity_field_is_checked(self):
        for field in IDENTITY:
            changed = [dict(row) for row in self.rows]
            changed[0][field] += "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_identity(self.rows, changed)

    def test_reserved_and_observed_propagate_separately(self):
        records = {}
        master.add_record(records, "observed", "a" * 64)
        master.add_record(records, "derived", "b" * 64, [("byte", "a" * 64), ("declared_source_group", "s:g")])
        master.add_record(records, "same_group", "c" * 64, [("declared_source_group", "s:g")])
        master.add_record(records, "reserved", "d" * 64)
        closed = master.exclusion_sets(records, [], {"reserved"}, {"observed"})
        self.assertEqual(closed["blocked_from_training"], ["reserved"])
        self.assertEqual(closed["excluded_from_final_holdout"], ["derived", "observed", "same_group"])


class HistoricalRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = getattr(audit, "OUTPUT", audit.HERE)
        cls.hashes = set((output / "final-exclusion-hashes.txt").read_text().splitlines())
        cls.closure = json.loads((output / "history-exposure-closure.json").read_text())

    def table(self, relative):
        with (audit.ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_all_visually_reviewed_candidates_are_final_excluded(self):
        rows = self.table(audit.D + "dinov3-colab-v2/audit-photos/rows.csv")
        expected = {r[k] for r in rows for k in ("sha256", "original_sha256", "normalized_sha256")}
        self.assertEqual(len(expected - self.hashes), 0, "Reviewed digests missing")

    def test_benchmark_scene_matches_reach_final_exclusion(self):
        geometry = self.table(audit.D + "dinov3-colab-v2/holdout-geometry-exclusions.csv")
        matches = self.table(audit.D + "dinov2-next-test-v1/taiwan-plantseg-scene-matches.csv")
        intake = self.table(audit.D + "dinov2-next-test-v1/plantseg-test-intake.csv")
        by_member = {r["archive_member"]: r["sha256"] for r in intake}
        expected = {r["original_sha256"] for r in geometry} | {by_member[r["new_member"]] for r in matches}
        self.assertEqual(len(expected - self.hashes), 0, "Known scene-match digests missing")
        self.assertGreater(len(self.closure.get("relation_evidence", [])), 0)

    def test_training_source_groups_protect_remaining_intake(self):
        train = self.table(audit.D + "dinov3-colab-v2/train-val.csv")
        groups = {r["group"] for r in train if r["source"] == "plantseg_v7"}
        intake = self.table(audit.D + "dinov2-next-test-v1/plantseg-test-intake.csv")
        expected = {r["sha256"] for r in intake if r["photo_source_url"] and r["source"] + ":" + r["photo_source_url"] in groups}
        self.assertEqual(len(expected - self.hashes), 0, "Same-source-group digests missing")


if __name__ == "__main__":
    unittest.main()
