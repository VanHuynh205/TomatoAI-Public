"""Replay only DINOv3 sampler indices and audit the sealed manifests."""

import ast
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.utils.data import WeightedRandomSampler

OUT = Path(__file__).resolve().parent
DERIVED = OUT.parent
TRAINER = "ai/dinov3_field.py"
ARCHIVES = {
    "v1": (
        "dinov3-colab-input.zip",
        "df6292e93d7d14a07b8e8946c0c4fe36959200d35c5b2d77685b075f6d2b64f9",
    ),
    "v2": (
        "dinov3-v2-package.zip",
        "a76e03fce4e9830148fcd9801ae32746fa51eb97bec460a5dd952b1a7c6cacbe",
    ),
}
V2_TRAIN_RESULTS_SHA = (
    "4254dbb58a5b17e805700f36ef9bfab9def7e1267d6148426cc6ef39801b5dee"
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_locked(version):
    folder = DERIVED / f"dinov3-colab-{version}"
    archive, archive_sha = ARCHIVES[version]
    assert file_sha(folder / archive) == archive_sha
    frozen = json.loads((folder / "frozen-model.json").read_bytes())
    history_bytes = (folder / "history.json").read_bytes()
    with ZipFile(folder / archive) as locked:
        trainer = locked.read(TRAINER)
        if version == "v1":
            receipt = json.loads(locked.read("input-receipt.json"))
            manifest = locked.read("data/train-val.csv")
            parameters = receipt["parameters"]
            assert sha(manifest) == receipt["train_val_sha256"]
            assert (
                sha(locked.read("holdout-lock.json")) == frozen["holdout_lock_sha256"]
            )
        else:
            manifest = locked.read("train-val.csv")
            plan = locked.read("training-plan.json")
            parameters = json.loads(plan)
            assert sha(plan) == frozen["training_plan_sha256"]
            assert sha(manifest) == frozen["train_val_sha256"]
            assert sha(trainer) == frozen["trainer_sha256"]
            assert sha(locked.read("run_dinov3_v2.py")) == frozen["runner_sha256"]
            assert (
                locked.read("v1-frozen-model.json")
                == (folder.parent / "dinov3-colab-v1/frozen-model.json").read_bytes()
            )
            results = folder / "dinov3-v2-train-results.zip"
            assert file_sha(results) == V2_TRAIN_RESULTS_SHA
            with ZipFile(results) as output:
                assert history_bytes == output.read("history.json")
                assert (folder / "frozen-model.json").read_bytes() == output.read(
                    "frozen-model.json"
                )
    assert manifest == (folder / "train-val.csv").read_bytes()
    history = json.loads(history_bytes)
    assert (
        max(history, key=lambda row: row["field"]["macro_f1"])["epoch"]
        == frozen["selected_epoch"]
        == 5
    )
    rows = list(csv.DictReader(io.StringIO(manifest.decode("utf-8"), newline="")))
    assert len({row["path"] for row in rows}) == len(rows)
    return (
        rows,
        trainer,
        history,
        {
            "package_sha256": archive_sha,
            "manifest_sha256": sha(manifest),
            "trainer_sha256": sha(trainer),
            "history_sha256": sha(history_bytes),
            "frozen_receipt_sha256": file_sha(folder / "frozen-model.json"),
            "original_torch": frozen["torch"],
            "selected_epoch": frozen["selected_epoch"],
            "parameters": parameters,
        },
    )


def probabilities(weights):
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def dimensions(row, additions):
    label, source, domain, group = (
        row[field] for field in ("class_id", "source", "domain", "group")
    )
    added = row["path"] in additions
    memberships = (
        ("all", "all"),
        ("class", label),
        ("domain", domain),
        ("source", source),
        ("source_class", f"{source}|{label}"),
        ("class_domain", f"{label}|{domain}"),
        ("source_domain", f"{source}|{domain}"),
        ("source_class_domain", f"{source}|{label}|{domain}"),
        ("group", f"{source}|{group}"),
        ("cohort", "added" if added else "original"),
    )
    return (
        (*memberships, ("added_source_class", f"{source}|{label}"))
        if added
        else memberships
    )


def concentration(values):
    shares = sorted(probabilities(values), reverse=True)
    return {
        "effective_count": 1 / math.fsum(share**2 for share in shares),
        "top1_share": shares[0],
        "top5_share": math.fsum(shares[:5]),
        "top10_share": math.fsum(shares[:10]),
    }


def validation_coverage(train, val, replay, history, selected):
    report = {}
    for scope, val_subset in [
        ("all", val),
        ("field", [row for row in val if row["domain"] == "field"]),
    ]:
        assert all(item[scope]["n"] == len(val_subset) for item in history)
        eligible = [
            i
            for i, row in enumerate(train)
            if scope == "all" or row["domain"] == "field"
        ]
        train_subset = [train[i] for i in eligible]
        scope_report = {}
        for name, fields in {
            "class": ("class_id",),
            "source": ("source",),
            "source_class": ("source", "class_id"),
            "source_group": ("source", "group"),
        }.items():

            def key(row):
                return "|".join(row[field] for field in fields)

            val_keys = {key(row) for row in val_subset}
            train_keys = {key(row) for row in train_subset}
            replay_keys = {key(train[i]) for i in eligible if replay[i]}
            scope_report[name] = {
                "train_keys": len(train_keys),
                "validation_keys": len(val_keys),
                "shared_keys": len(train_keys & val_keys),
                "replayed_train_keys": len(replay_keys),
                "shared_replayed_keys": len(replay_keys & val_keys),
                "validation_only": sorted(val_keys - train_keys),
                "train_only": sorted(train_keys - val_keys)
                if name != "source_group"
                else None,
                "train_replay_share_covered": sum(
                    replay[i] for i in eligible if key(train[i]) in val_keys
                )
                / sum(replay[i] for i in eligible),
            }
        scope_report["rows"] = len(val_subset)
        scope_report["selected_per_class"] = selected[scope]["per_class"]
        report[scope] = scope_report
    return report


def exposure(train, val, weights, provenance, history, additions):
    samples, seed = (provenance["parameters"][key] for key in ("samples", "seed"))
    epochs = provenance["selected_epoch"]
    p = probabilities(weights)
    class_mass = Counter()
    for row, mass in zip(train, p, strict=True):
        class_mass[row["class_id"]] += mass
    assert len(class_mass) == 9 and all(
        math.isclose(mass, 1 / 9) for mass in class_mass.values()
    )

    draw_counts = Counter()
    epoch_stats = []
    for epoch in range(epochs):
        sampler = WeightedRandomSampler(
            weights,
            samples,
            replacement=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        assert (
            sampler.weights.dtype == torch.float64
            and sampler.weights.device.type == "cpu"
        )
        indices = list(sampler)
        assert len(indices) == samples and all(0 <= i < len(train) for i in indices)
        draw_counts.update(indices)
        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "seed": seed + epoch,
                "unique_images": len(set(indices)),
                "repeated_draws": samples - len(set(indices)),
                "added_draws": sum(train[i]["path"] in additions for i in indices),
                "ordered_indices_json_sha256": sha(
                    json.dumps(indices, separators=(",", ":")).encode()
                ),
            }
        )
    total = samples * epochs
    assert draw_counts.total() == total
    mass_by_group, replay_by_group = Counter(), Counter()
    items = {}

    def entry(dimension, key):
        return items.setdefault(
            (dimension, key),
            {
                "dimension": dimension,
                "key": key,
                "train_rows": 0,
                "validation_rows": 0,
                "expected_draws_per_epoch": 0.0,
                "expected_unique_images_selected": 0.0,
                "replay_draws_selected": 0,
                "replay_unique_images_selected": 0,
            },
        )

    for i, (row, mass) in enumerate(zip(train, p, strict=True)):
        group = (row["source"], row["group"])
        mass_by_group[group] += mass
        replay_by_group[group] += draw_counts[i]
        unique_probability = -math.expm1(total * math.log1p(-mass))
        for dimension, key in dimensions(row, additions):
            item = entry(dimension, key)
            item["train_rows"] += 1
            item["expected_draws_per_epoch"] += mass * samples
            item["expected_unique_images_selected"] += unique_probability
            item["replay_draws_selected"] += draw_counts[i]
            item["replay_unique_images_selected"] += int(draw_counts[i] > 0)
    for row in val:
        for dimension, key in dimensions(row, additions):
            entry(dimension, key)["validation_rows"] += 1
    for item in items.values():
        item["expected_share"] = item["expected_draws_per_epoch"] / samples
        item["expected_draws_selected"] = item["expected_draws_per_epoch"] * epochs
        item["replay_share_selected"] = item["replay_draws_selected"] / total
        item["replay_repeated_draws_selected"] = (
            item["replay_draws_selected"] - item["replay_unique_images_selected"]
        )
    assert items["all", "all"]["replay_unique_images_selected"] == len(draw_counts)
    assert items["all", "all"]["validation_rows"] == len(val)
    selected = next(row for row in history if row["epoch"] == epochs)
    return {
        "provenance": provenance,
        "epochs": epoch_stats,
        "coverage": validation_coverage(train, val, draw_counts, history, selected),
        "selected_validation_macro_f1": {
            scope: selected[scope]["macro_f1"] for scope in ("field", "all")
        },
        "image_concentration": {
            "expected": concentration(p),
            "replay": concentration(draw_counts.values()),
            "replay_max_draws_per_image": max(draw_counts.values()),
        },
        "group_concentration": {
            "expected": concentration(mass_by_group.values()),
            "replay": concentration(replay_by_group.values()),
            "train_groups": len(mass_by_group),
            "replay_groups": sum(count > 0 for count in replay_by_group.values()),
            "largest_expected": [
                {"source": k[0], "group": k[1], "share": v}
                for k, v in mass_by_group.most_common(5)
            ],
            "largest_replay": [
                {"source": k[0], "group": k[1], "draws": v}
                for k, v in replay_by_group.most_common(5)
            ],
        },
        "exposure": [items[key] for key in sorted(items)],
    }


def compare_expected(original, expanded, old_weights, new_weights, budget, additions):
    scenarios = {}
    for name, rows, weights in (
        ("old_policy_original", original, old_weights(original)),
        ("old_policy_expanded", expanded, old_weights(expanded)),
        ("new_policy_original", original, new_weights(original, budget)),
        ("new_policy_expanded", expanded, new_weights(expanded, budget)),
    ):
        shares = {
            dimension: Counter()
            for dimension in (
                "class",
                "domain",
                "source",
                "source_class",
                "class_domain",
                "cohort",
                "image",
            )
        }
        for row, mass in zip(rows, probabilities(weights), strict=True):
            shares["image"][row["path"]] = mass
            for dimension, key in dimensions(row, additions):
                if dimension in shares:
                    shares[dimension][key] += mass
        scenarios[name] = shares
    distances = {}
    for name, start, end in (
        ("policy_on_original", "old_policy_original", "new_policy_original"),
        ("policy_on_expanded", "old_policy_expanded", "new_policy_expanded"),
        ("additions_under_old_policy", "old_policy_original", "old_policy_expanded"),
        ("additions_under_new_policy", "new_policy_original", "new_policy_expanded"),
        ("combined", "old_policy_original", "new_policy_expanded"),
    ):
        before, after = scenarios[start], scenarios[end]
        distances[name] = {
            dimension: math.fsum(
                abs(before[dimension][key] - after[dimension][key])
                for key in before[dimension].keys() | after[dimension].keys()
            )
            / 2
            for dimension in before
        }
    for shares in scenarios.values():
        del shares["image"]
    return {
        "draws_per_epoch": budget,
        "total_variation": distances,
        "scenario_expected_shares": scenarios,
    }


def main():
    rows1, trainer1, history1, facts1 = read_locked("v1")
    rows2, trainer2, history2, facts2 = read_locked("v2")
    root = DERIVED.parents[1]
    assert trainer2 == (root / TRAINER).read_bytes()

    def old_policy_ast(trainer):
        function = next(
            node
            for node in ast.parse(trainer).body
            if isinstance(node, ast.FunctionDef) and node.name == "sampler_weights"
        )
        return ast.dump(function)

    assert old_policy_ast(trainer1) == old_policy_ast(trainer2)
    sys.path.insert(0, str(root))
    from ai.dinov3_field import continuation_weights, sampler_weights

    assert rows1 == rows2[: len(rows1)]
    extra = rows2[len(rows1) :]
    assert len(extra) == 288 and all(row["split"] == "train" for row in extra)
    assert {row["domain"] for row in extra} == {"mixed", "controlled"}
    additions = {row["path"] for row in extra}
    train1, val1 = (
        [row for row in rows1 if row["split"] == split] for split in ("train", "val")
    )
    train2, val2 = (
        [row for row in rows2 if row["split"] == split] for split in ("train", "val")
    )
    assert (len(train1), len(train2), len(val1)) == (7781, 8069, 1642) and val1 == val2
    for train in (train1, train2):
        assert not ({row["group"] for row in train} & {row["group"] for row in val1})
    assert {scope: history2[0][scope] for scope in ("field", "all")} == {
        scope: history1[4][scope] for scope in ("field", "all")
    }
    assert [
        (facts["parameters"]["samples"], facts["parameters"]["seed"])
        for facts in (facts1, facts2)
    ] == [(3072, 42), (4096, 1729)]
    budget = facts2["parameters"]["samples"]
    audit = {
        "replay": {
            "local_torch": str(torch.__version__),
            "device": "cpu",
            "weight_dtype": "float64",
            "replacement": True,
            "seed": "base seed + zero-based epoch",
            "v2_epoch_zero_draws": 0,
            "scope": "Deterministic sampler index replay with local torch. Original Colab indices were not logged. Augmentation, model inference and gradients are outside the audit.",
        },
        "v2_training_results_archive_sha256": V2_TRAIN_RESULTS_SHA,
        "unchanged_validation_manifest": True,
        "v1_sampler_ast_matches_v2": True,
        "v2_trainer_bytes_match_workspace": True,
        "expected_counterfactuals": compare_expected(
            train1, train2, sampler_weights, continuation_weights, budget, additions
        ),
        "v1": exposure(
            train1, val1, sampler_weights(train1), facts1, history1, additions
        ),
        "v2": exposure(
            train2,
            val2,
            continuation_weights(train2, budget),
            facts2,
            history2,
            additions,
        ),
    }
    with (OUT / "sampling-exposure.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", *audit["v1"]["exposure"][0]])
        writer.writeheader()
        for version in ("v1", "v2"):
            writer.writerows(
                {"run": version, **item} for item in audit[version]["exposure"]
            )
            audit[version]["exposure"] = [
                item
                for item in audit[version]["exposure"]
                if item["dimension"] != "group"
            ]
    audit["group_exposure_detail"] = "sampling-exposure.csv"
    (OUT / "sampling-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
