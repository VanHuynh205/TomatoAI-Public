"""Task 6 / P09: Export YOLO26s-cls CPU bundles under Amendment v3, verify parity and benchmark p95."""

import hashlib
import io
import json
import os
import secrets
import tempfile
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from ai.artifacts import _WEIGHTS_LIMIT, _local, _name, _read, _threshold, load_bundle
from ai.benchmark import benchmark, read_benchmark_images
from ai.data import ManifestDataset, _sha256
from ai.inference import Predictor
from ai.models import build_model
from ai.preprocessing import build_transform
from ai.train import _atomic_save
from desktop.tests.browser_server import ROOT, create_browser_app
from desktop.tests.real_smoke import assert_parity

from .head_recovery import (  # type: ignore[import-not-found]
    HERE,
    VENDOR_CHECK,
    select_candidate,
    summarize,
)

V3_GATE = {
    "plantdoc_supported_macro_f1_strictly_greater_than": 0.40,
    "pv_macro_f1_minimum": 0.9780424202954058,
}


def eligible_candidates(
    variants: dict[str, Any], gate: dict[str, float] = V3_GATE
) -> list[str]:
    return [
        name
        for name, m in variants.items()
        if m["plantdoc_f1"] > gate["plantdoc_supported_macro_f1_strictly_greater_than"]
        and m["pv_f1"] >= gate["pv_macro_f1_minimum"]
        and m["pooled_f1"] >= 0.85
        and m["min_recall"] >= 0.70
        and m["threshold"] is not None
    ]


def select_v3_winner(
    variants: dict[str, Any], gate: dict[str, float] = V3_GATE
) -> str | None:
    winner: str | None = select_candidate(variants, gate)
    return winner


def write_yolo_bundle(
    *,
    weights_path: Path,
    decision_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    model_version: str,
    threshold: float | None,
) -> Path:
    weights_file = _local(weights_path)
    decision_file = _local(decision_path)
    output = _local(output_dir)
    if output.exists():
        raise FileExistsError(f"bundle already exists: {output}")
    if not _name(model_version) or not _threshold(threshold):
        raise ValueError("invalid model_version or threshold")
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    decision_bytes = _read(decision_file, 1_000_000)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "architecture": config["architecture"],
        "model_version": model_version,
        "class_ids": config["class_ids"],
        "preprocessing": config["preprocessing"],
        "dataset_version": Path(config["dataset_dir"]).name,
        "dataset_sha256": config["seal_sha256"],
        "split_hash": config["split_hash"],
        "threshold": threshold,
        "class_map_sha256": hashlib.sha256(
            json.dumps(
                config["class_ids"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "validation_decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "artifact_kind": "trained",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        torch.save(state, stage / "model.pth")
        meta["weights_sha256"] = hashlib.sha256(
            _read(stage / "model.pth", _WEIGHTS_LIMIT)
        ).hexdigest()
        (stage / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        load_bundle(stage, require_trained=True)
        if output.exists():
            raise FileExistsError(f"bundle already exists: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            for name in ("model.pth", "metadata.json"):
                (stage / name).unlink(missing_ok=True)
            stage.rmdir()
    return output


def verify_bundle_parity(
    *,
    run_label: str,
    bundle_path: Path,
    weights_path: Path,
    decision_path: Path,
    logits_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    bundle = load_bundle(bundle_path, require_trained=True)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert bundle.metadata["preprocessing"] == config["preprocessing"]
    assert bundle.metadata["class_ids"] == config["class_ids"]
    assert bundle.metadata["dataset_sha256"] == config["seal_sha256"]
    assert bundle.metadata["validation_decision_sha256"] == _sha256(decision_path)
    for key, tensor in bundle.model.state_dict().items():
        torch.testing.assert_close(tensor, state[key], rtol=0, atol=0)

    reference = build_model(
        config["architecture"], len(config["class_ids"]), None
    ).eval()
    reference.load_state_dict(state, strict=True)
    dataset = ManifestDataset(
        Path(config["dataset_dir"]),
        "val",
        build_transform(config["preprocessing"]["spec"], training=False),
    )
    saved_logits = json.loads(logits_path.read_text(encoding="utf-8-sig"))
    probs = (
        torch.tensor(saved_logits["logits"]).softmax(dim=1).max(dim=1).values.tolist()
    )
    scores = dict(zip(saved_logits["ids"], probs, strict=True))

    selected: set[int] = set()
    for prefix in ("pv:", "plantdoc:"):
        source_indices = [
            i
            for i, row in enumerate(dataset._rows)
            if row["sample_id"].startswith(prefix)
        ]
        for label in config["class_ids"]:
            matches = [
                i for i in source_indices if dataset._rows[i]["class_id"] == label
            ]
            if matches:
                selected.add(matches[0])
        by_score = sorted(
            source_indices, key=lambda i: scores[dataset._rows[i]["sample_id"]]
        )
        selected.update((by_score[0], by_score[-1]))

    predictor = Predictor(bundle_path)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(_module: Any, args: Any, logits: Any) -> None:
        assert torch.is_inference_mode_enabled()
        captured.append((args[0].clone(), logits.clone()))

    handle = predictor._model.register_forward_hook(capture)
    os.environ["TOMATO_E2E_MODEL_DIR"] = str(bundle_path)
    os.environ["TOMATO_E2E_CARE_PATH"] = str(ROOT / "infra/knowledge/care-v1.json")
    token, origin = secrets.token_urlsafe(32), "http://127.0.0.1:8765"
    app = create_browser_app("real", token, origin)
    protected = [
        weights_path,
        decision_path,
        bundle_path / "model.pth",
        bundle_path / "metadata.json",
        ROOT / "infra/knowledge/care-v1.json",
    ]
    before = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in protected}
    cases: list[dict[str, Any]] = []
    try:
        with TestClient(
            app,
            base_url=origin,
            headers={"X-Session-Token": token, "Origin": origin},
        ) as client:
            health = client.get("/api/v1/health")
            assert health.status_code == 200
            assert health.json()["model_version"] == bundle.metadata["model_version"]
            knowledge = health.json()["knowledge_version"]
            for index in sorted(selected):
                row = dataset._rows[index]
                assert row["split"] == "val"
                tensor, _target, sample_id = dataset[index]
                with torch.inference_mode():
                    logits = reference(tensor.unsqueeze(0))
                with Image.open(dataset.source_root / row["source_path"]) as image:
                    expected = predictor.predict(image)
                    stream = io.BytesIO()
                    with (
                        ImageOps.exif_transpose(image) as oriented,
                        oriented.convert("RGB") as rgb,
                    ):
                        rgb.save(stream, format="PNG")
                observed_tensor, observed_logits = captured.pop()
                torch.testing.assert_close(
                    observed_tensor, tensor.unsqueeze(0), rtol=0, atol=1e-6
                )
                torch.testing.assert_close(observed_logits, logits, rtol=0, atol=1e-6)
                max_err = float((observed_logits - logits).abs().max().item())
                assert max_err <= 1e-5
                response = client.post(
                    "/api/v1/predict",
                    files={"image": ("validation.png", stream.getvalue(), "image/png")},
                )
                assert response.status_code == 200, response.text
                assert_parity(expected, response.json(), knowledge)
                cases.append(
                    {
                        "sample_id": sample_id,
                        "split": "val",
                        "prediction": asdict(expected),
                        "logit_max_abs_error": max_err,
                    }
                )
    finally:
        handle.remove()
    assert before == {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in protected
    }
    assert {case["prediction"]["status"] for case in cases} == {"ok", "uncertain"}
    return {
        "run": run_label,
        "bundle": bundle.metadata["model_version"],
        "input_sha256": before,
        "state_dict_tensors_exact": len(state),
        "cases": cases,
    }


def load_run_metric(run_id: str, class_ids: list[str]) -> dict[str, Any]:
    pred_path = ROOT / "ai/runs" / run_id / "validation-predictions.json"
    payload = json.loads(pred_path.read_text(encoding="utf-8-sig"))
    ids = [r["sample_id"] for r in payload["predictions"]]
    labels = torch.tensor([r["target_index"] for r in payload["predictions"]])
    logits = torch.tensor([r["logits"] for r in payload["predictions"]])
    metric: dict[str, Any] = summarize(
        {"ids": ids, "labels": labels}, logits, class_ids
    )
    return metric


def main() -> None:
    check = import_module(VENDOR_CHECK)
    lock_path = HERE / "p09-lock.json"
    lock = check.read_json(lock_path)
    for name, digest in lock["files_sha256"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(f"P09 input hash mismatch: {name}")
    config = check.read_json(ROOT / "ai/configs/E10.json")
    class_ids: list[str] = config["class_ids"]

    p07_dec = check.read_json(HERE / "P07/decision.json")
    p08_dec = check.read_json(HERE / "P08/decision.json")
    variants = {
        "E09": load_run_metric("E09", class_ids),
        "E10": load_run_metric("E10", class_ids),
        "p07-refit": p07_dec["metric"],
        "p08-refit": p08_dec["metric"],
    }
    eligible = eligible_candidates(variants, V3_GATE)
    winner = select_v3_winner(variants, V3_GATE)
    if eligible != ["p07-refit", "p08-refit"] or winner != "p07-refit":
        raise ValueError(
            f"Unexpected v3 selection: eligible={eligible}, winner={winner}"
        )

    output = HERE / "P09"
    output.mkdir(exist_ok=False)
    decision_path = output / "decision.json"
    _atomic_save(
        decision_path,
        {
            "schema_version": 1,
            "protocol_id": "yolo-image-P02-validation-only-v3",
            "amendment_path": "docs/progress/yolo-image/protocol-amendment-v3.md",
            "lock_sha256": _sha256(lock_path),
            "gate": V3_GATE,
            "eligible_candidates": eligible,
            "winner": winner,
            "companion_candidate": "p08-refit",
            "winner_weights_sha256": p07_dec["weights_sha256"],
            "winner_logits_sha256": p07_dec["logits_sha256"],
            "companion_weights_sha256": p08_dec["weights_sha256"],
            "companion_logits_sha256": p08_dec["logits_sha256"],
            "app_rollback_bundle": "b0-v1",
            "test_opened": False,
        },
    )

    torch.set_num_threads(4)
    v1_bundle = ROOT / "ai/artifacts/yolo26s-cls-v1"
    p08_bundle = ROOT / "ai/artifacts/yolo26s-cls-p08-v1"
    write_yolo_bundle(
        weights_path=HERE / "P07/refit.pth",
        decision_path=decision_path,
        config=config,
        output_dir=v1_bundle,
        model_version="yolo26s-cls-v1",
        threshold=p07_dec["metric"]["threshold"],
    )
    write_yolo_bundle(
        weights_path=HERE / "P08/refit.pth",
        decision_path=HERE / "P08/decision.json",
        config=config,
        output_dir=p08_bundle,
        model_version="yolo26s-cls-p08-v1",
        threshold=p08_dec["metric"]["threshold"],
    )

    parity_results = [
        verify_bundle_parity(
            run_label="P07",
            bundle_path=v1_bundle,
            weights_path=HERE / "P07/refit.pth",
            decision_path=decision_path,
            logits_path=HERE / "P07/p02-validation-logits.json",
            config=config,
        ),
        verify_bundle_parity(
            run_label="P08",
            bundle_path=p08_bundle,
            weights_path=HERE / "P08/refit.pth",
            decision_path=HERE / "P08/decision.json",
            logits_path=HERE / "P08/p02-validation-logits.json",
            config=config,
        ),
    ]
    _atomic_save(
        output / "candidate-parity.json",
        {
            "status": "passed",
            "scope": "Validation-only checkpoint/CPU/production-ASGI parity for YOLO26s-cls bundles; no live TCP, new field TEST or deployment claim.",
            "decision_sha256": _sha256(decision_path),
            "results": parity_results,
        },
    )

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(6)
    except RuntimeError:
        pass
    manifest_path = ROOT / "docs/experiments/benchmark-images.json"
    images = read_benchmark_images(manifest_path)
    bench = benchmark(v1_bundle, images, repeats=100)
    bench["images_manifest_sha256"] = _sha256(manifest_path)
    if bench["warm_p95_ms"] > 3000.0:
        raise ValueError(f"YOLO CPU warm p95 exceeded 3000 ms: {bench['warm_p95_ms']}")
    bench_out = ROOT / "docs/experiments/benchmark-yolo26s-cls-v1.json"
    _atomic_save(bench_out, dict(sorted(bench.items())))
    print(
        json.dumps(
            {
                "status": "passed",
                "bundles": ["yolo26s-cls-v1", "yolo26s-cls-p08-v1"],
                "parity_cases": [len(r["cases"]) for r in parity_results],
                "cold_load_ms": round(bench["cold_load_ms"], 2),
                "warm_p50_ms": round(bench["warm_p50_ms"], 2),
                "warm_p95_ms": round(bench["warm_p95_ms"], 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    main()
