"""Verify new research bundles against CPU checkpoints and the production ASGI app."""

import io
import json
import os
import secrets
from dataclasses import asdict
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from ai.artifacts import load_bundle
from ai.data import ManifestDataset, _sha256
from ai.inference import Predictor
from ai.models import build_model
from ai.preprocessing import build_transform
from desktop.tests.browser_server import create_browser_app
from desktop.tests.real_smoke import assert_parity

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).parent


def main():
    torch.set_num_threads(4)
    summary_path = ROOT / "docs/experiments/validation-summary-adaptation-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    for architecture, version in (
        ("efficientnet_b0", "b0-adapt-v1"),
        ("resnet50", "resnet50-adapt-v1"),
    ):
        run = ROOT / "ai/runs" / summary["selected_by_architecture"][architecture]
        bundle_path = ROOT / "ai/artifacts" / version
        bundle = load_bundle(bundle_path, require_trained=True)
        best = torch.load(run / "best.pth", map_location="cpu", weights_only=True)
        config = best["config"]
        assert bundle.metadata["preprocessing"] == config["preprocessing"]
        assert bundle.metadata["class_ids"] == config["class_ids"]
        assert bundle.metadata["dataset_sha256"] == config["seal_sha256"]
        assert bundle.metadata["validation_decision_sha256"] == _sha256(
            run / "decision.json"
        )
        for key, tensor in bundle.model.state_dict().items():
            torch.testing.assert_close(tensor, best["model"][key], rtol=0, atol=0)
        reference = build_model(architecture, len(config["class_ids"])).eval()
        reference.load_state_dict(best["model"], strict=True)
        dataset = ManifestDataset(
            Path(config["dataset_dir"]),
            "val",
            build_transform(config["preprocessing"]["spec"], training=False),
        )
        saved = json.loads((run / "validation-predictions.json").read_text())
        scores = {
            row["sample_id"]: max(row["probabilities"]) for row in saved["predictions"]
        }
        selected = set()
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
        captured = []

        def capture(_module, args, logits, captured=captured):
            assert torch.is_inference_mode_enabled()
            captured.append((args[0].clone(), logits.clone()))

        handle = predictor._model.register_forward_hook(capture)
        os.environ["TOMATO_E2E_MODEL_DIR"] = str(bundle_path)
        os.environ["TOMATO_E2E_CARE_PATH"] = str(ROOT / "infra/knowledge/care-v1.json")
        token, origin = secrets.token_urlsafe(32), "http://127.0.0.1:8765"
        app = create_browser_app("real", token, origin)
        protected = [
            run / "best.pth",
            run / "decision.json",
            bundle_path / "model.pth",
            bundle_path / "metadata.json",
            ROOT / "infra/knowledge/care-v1.json",
        ]
        before = {str(path.relative_to(ROOT)): _sha256(path) for path in protected}
        cases = []
        try:
            with TestClient(
                app,
                base_url=origin,
                headers={"X-Session-Token": token, "Origin": origin},
            ) as client:
                health = client.get("/api/v1/health")
                assert health.status_code == 200
                assert health.json()["model_version"] == version
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
                    torch.testing.assert_close(
                        observed_logits, logits, rtol=0, atol=1e-6
                    )
                    response = client.post(
                        "/api/v1/predict",
                        files={
                            "image": ("validation.png", stream.getvalue(), "image/png")
                        },
                    )
                    assert response.status_code == 200, response.text
                    assert_parity(expected, response.json(), knowledge)
                    cases.append(
                        {
                            "sample_id": sample_id,
                            "split": "val",
                            "prediction": asdict(expected),
                            "logit_max_abs_error": (observed_logits - logits)
                            .abs()
                            .max()
                            .item(),
                        }
                    )
        finally:
            handle.remove()
        assert before == {
            str(path.relative_to(ROOT)): _sha256(path) for path in protected
        }
        assert {case["prediction"]["status"] for case in cases} == {"ok", "uncertain"}
        results.append(
            {
                "run": run.name,
                "bundle": version,
                "input_sha256": before,
                "state_dict_tensors_exact": len(best["model"]),
                "cases": cases,
            }
        )
    output = {
        "status": "passed",
        "scope": "Validation-only checkpoint/CPU/production-ASGI parity; no live TCP, new field TEST or deployment claim.",
        "summary_sha256": _sha256(summary_path),
        "results": results,
    }
    (REPORT / "candidate-parity.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "cases": [len(r["cases"]) for r in results]}))


if __name__ == "__main__":
    main()
