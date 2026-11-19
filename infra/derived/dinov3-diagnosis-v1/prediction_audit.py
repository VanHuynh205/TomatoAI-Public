"""Audit saved DINOv3 predictions; never load a model or select a threshold."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

OUT = Path(__file__).resolve().parent
DERIVED = OUT.parent
IDENTITY = ("path", "sha256", "class_id", "source", "domain", "group")
NEGATIVE = "not_tomato_leaf"
THRESHOLD = 0.5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(row):
    return row["prediction"] != NEGATIVE and float(row["confidence"]) >= THRESHOLD


def correct(row):
    return row["prediction"] == row["class_id"]


def pair_rows(before, after):
    old = {row["path"]: row for row in before}
    new = {row["path"]: row for row in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError("Duplicate paths or different image populations")
    pairs = []
    for path in sorted(old):
        a, b = old[path], new[path]
        if any(a[key] != b[key] for key in IDENTITY):
            raise ValueError(f"Identity mismatch: {path}")
        for row in (a, b):
            confidence = float(row["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence: {path}")
            if row["status"] != ("ok" if accepted(row) else "uncertain"):
                raise ValueError(f"Stored status differs from fixed policy: {path}")
        pairs.append((a, b))
    return pairs


def summary(rows):
    chosen = [row for row in rows if accepted(row)]
    wrong = [row for row in rows if not correct(row)]
    bacterial = [row for row in rows if row["prediction"] == "bacterial_spot"]
    return {
        "n": len(rows),
        "correct": sum(map(correct, rows)),
        "accepted": len(chosen),
        "accepted_correct": sum(map(correct, chosen)),
        "accepted_wrong": sum(not correct(row) for row in chosen),
        "healthy_accepted_as_disease": sum(
            row["class_id"] == "healthy" and row["prediction"] != "healthy"
            for row in chosen
        ),
        "wrong_with_confidence_ge_0_7": sum(float(row["confidence"]) >= 0.7 for row in wrong),
        "mean_confidence": fmean(float(row["confidence"]) for row in rows) if rows else None,
        "mean_wrong_confidence": (
            fmean(float(row["confidence"]) for row in wrong) if wrong else None
        ),
        "bacterial_spot_predictions": {
            "total": len(bacterial),
            "correct": sum(map(correct, bacterial)),
            "accepted": sum(map(accepted, bacterial)),
            "accepted_correct": sum(correct(row) and accepted(row) for row in bacterial),
            "true_labels": dict(Counter(row["class_id"] for row in bacterial)),
        },
    }


def paired_summary(pairs):
    before, after = ([pair[i] for pair in pairs] for i in (0, 1))
    regressed = [(a, b) for a, b in pairs if correct(a) and not correct(b)]
    corrected = [(a, b) for a, b in pairs if not correct(a) and correct(b)]
    a, b = summary(before), summary(after)
    assert b["correct"] - a["correct"] == len(corrected) - len(regressed)
    return {
        "v1": a,
        "v2": b,
        "regressed": len(regressed),
        "corrected": len(corrected),
        "regression_destinations": dict(Counter(b["prediction"] for _, b in regressed)),
        "new_accepted_wrong": sum(accepted(b) and not correct(b) and
                                  not (accepted(a) and not correct(a)) for a, b in pairs),
        "resolved_accepted_wrong": sum(accepted(a) and not correct(a) and
                                       not (accepted(b) and not correct(b)) for a, b in pairs),
        "acceptance_transitions": dict(Counter(
            f"{a['status']} -> {b['status']}" for a, b in pairs
        )),
    }


def self_check():
    base = dict(zip(IDENTITY, ["a", "hash", "healthy", "s", "field", "g"]))
    a = {**base, "prediction": "healthy", "confidence": "0.5", "status": "ok"}
    b = {**a, "prediction": "bacterial_spot", "confidence": "0.8"}
    c = {**a, "path": "b", "prediction": NEGATIVE, "confidence": "0.99", "status": "uncertain"}
    d = {**a, "path": "b", "confidence": "0.49", "status": "uncertain"}
    pairs = pair_rows([a, c], [d, b])
    result = paired_summary(pairs)
    assert result["regressed"] == result["corrected"] == 1
    assert result["v2"]["accepted_wrong"] == result["v2"]["healthy_accepted_as_disease"] == 1
    assert result["new_accepted_wrong"] == 1 and not accepted(c) and not accepted(d)
    for left, right in [([a, a], [b, b]), ([a], [{**b, "group": "other"}]),
                        ([a], [{**b, "confidence": "nan"}]),
                        ([a], [{**b, "status": "uncertain"}]),
                        ([a], [{**a, "confidence": "0.2", "status": "invalid"}])]:
        try:
            pair_rows(left, right)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed input was accepted")


def main():
    self_check()
    comparison_path = DERIVED / "dinov3-colab-v2/comparison.json"
    comparison = json.loads(comparison_path.read_bytes())
    assert comparison["acceptance_threshold"] == THRESHOLD
    report = {"scope": "Saved predictions only; historical tests and SLIF are observed.",
              "threshold": THRESHOLD, "comparison_sha256": digest(comparison_path),
              "benchmarks": {}, "representative_regressions": []}
    transitions = []
    for benchmark, expected_n in [("legacy", 2647), ("external", 460), ("slif", 21)]:
        paths = {
            "v1": DERIVED / (f"dinov3-colab-v1/dinov3-{benchmark}-predictions.csv"
                             if benchmark != "slif" else
                             "dinov3-colab-v2/dinov3-v1-slif-predictions.csv"),
            "v2": DERIVED / f"dinov3-colab-v2/dinov3-v2-{benchmark}-predictions.csv",
        }
        loaded = {}
        for version, path in paths.items():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                loaded[version] = list(csv.DictReader(stream))
        pairs = pair_rows(loaded["v1"], loaded["v2"])
        assert len(pairs) == expected_n
        source = (comparison["historical_benchmarks"][benchmark] if benchmark != "slif"
                  else comparison["slif_pilot"])
        for version, path in paths.items():
            assert digest(path) == source["predictions_sha256"][f"dinov3_{version}"]
        scopes = {"all_images": pairs, "positive_field": [
            pair for pair in pairs if pair[0]["domain"] == "field" and pair[0]["class_id"] != NEGATIVE
        ]}
        item = {"input_sha256": {version: digest(path) for version, path in paths.items()},
                "slices": {name: paired_summary(rows) for name, rows in scopes.items()},
                "by_source_domain_class": []}
        if benchmark != "slif":
            for name in scopes:
                for version in paths:
                    saved = source["models"][f"dinov3_{version}"][name]
                    observed = item["slices"][name][version]
                    assert observed["n"] == saved["top1"]["n"]
                    assert math.isclose(observed["correct"] / observed["n"], saved["top1"]["accuracy"])
                    assert observed["accepted"] == saved["acceptance"]["count"]
                    assert observed["accepted_correct"] == saved["accepted_accuracy"]["count"]
        groups = defaultdict(list)
        counts = Counter()
        for a, b in pairs:
            key = tuple(a[k] for k in ("source", "domain", "class_id"))
            groups[key].append((a, b))
            counts[(*key, a["prediction"], b["prediction"], a["status"], b["status"])] += 1
        for key, rows in sorted(groups.items()):
            item["by_source_domain_class"].append({
                **dict(zip(("source", "domain", "class_id"), key)), **paired_summary(rows)
            })
            regressed = [(a, b) for a, b in rows if correct(a) and not correct(b)]
            if regressed:
                a, b = max(regressed, key=lambda pair: float(pair[1]["confidence"]))
                report["representative_regressions"].append({
                    "benchmark": benchmark, **{k: a[k] for k in IDENTITY},
                    "v1_prediction": a["prediction"], "v1_confidence": float(a["confidence"]),
                    "v2_prediction": b["prediction"], "v2_confidence": float(b["confidence"]),
                    "selection": "highest v2 confidence among regressions in source/domain/class",
                })
        transitions.extend({"benchmark": benchmark,
                            **dict(zip(("source", "domain", "class_id", "v1_prediction",
                                        "v2_prediction", "v1_status", "v2_status"), key)),
                            "count": count} for key, count in sorted(counts.items()))
        report["benchmarks"][benchmark] = item
    assert sum(row["count"] for row in transitions) == 3128
    (OUT / "prediction-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "prediction-transitions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    print("Verified 3,128 paired rows, fixed-policy counts and input hashes; self-checks passed.")


if __name__ == "__main__":
    main()
