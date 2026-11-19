"""Regression checks for the Colab preflights using synthetic inputs."""

import ast
import json
import runpy
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "docs/progress/yolo26s-field-colab-2026-11-10"


def test_api_refuses_optimized_python() -> None:
    result = subprocess.run(
        [sys.executable, "-O", "-S", str(SCRIPTS / "colab_api_check.py")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "API checks require assertions enabled" in result.stderr
    assert "API_CHECK_OK" not in result.stdout


def test_baseline_child_refuses_optimized_python() -> None:
    tree = ast.parse((SCRIPTS / "colab_baseline_check.py").read_text(encoding="utf-8"))
    smoke = next(
        node.value.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "smoke" for target in node.targets)
    )
    result = subprocess.run(
        [sys.executable, "-O", "-S", "-c", textwrap.dedent(smoke)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Baseline smoke requires assertions enabled" in result.stderr
    assert "BASELINE_SMOKE_OK" not in result.stdout


def test_failed_baseline_upload_supersedes_a_prior_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    current = tmp_path / "baseline-smoke.json"
    legacy = '{"status": "synthetic_smoke_only_not_accuracy"}'
    current.write_text(legacy, encoding="utf-8")
    running: list[dict[str, str]] = []

    def upload() -> dict[str, bytes]:
        pointer = json.loads(current.read_text(encoding="utf-8"))
        report = tmp_path / pointer.get("report", current.name)
        running.append(json.loads(report.read_text(encoding="utf-8")))
        return {}

    google = ModuleType("google")
    colab = ModuleType("google.colab")
    monkeypatch.setattr(colab, "files", SimpleNamespace(upload=upload), raising=False)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    main = runpy.run_path(str(SCRIPTS / "colab_baseline_check.py"))["main"]
    monkeypatch.setitem(main.__globals__, "DRIVE", tmp_path)
    with pytest.raises(ValueError, match="Upload only baseline-upload.zip"):
        main()

    assert running[0]["status"] == "running"
    pointer = json.loads(current.read_text(encoding="utf-8"))
    report = tmp_path / pointer["report"]
    failed = json.loads(report.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["run_id"] == pointer["run_id"] == running[0]["run_id"]
    assert (report.parent / "previous-baseline-smoke.json").read_text(encoding="utf-8") == legacy
    assert "BASELINE_SMOKE_OK" not in capsys.readouterr().out


def test_older_baseline_success_cannot_replace_newer_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    main = runpy.run_path(str(SCRIPTS / "colab_baseline_check.py"))["main"]
    monkeypatch.setitem(main.__globals__, "DRIVE", tmp_path)
    current = tmp_path / "baseline-smoke.json"

    def run_smoke(run_dir: Path) -> dict[str, list[int]]:
        def fail_smoke(_run_dir: Path) -> None:
            raise RuntimeError("synthetic smoke error")

        monkeypatch.setitem(main.__globals__, "run_smoke", fail_smoke)
        with pytest.raises(RuntimeError, match="synthetic smoke error"):
            main()
        return {"class_ids": [0, 1, 2, 3, 4, 5, 6, 7]}

    monkeypatch.setitem(main.__globals__, "run_smoke", run_smoke)
    main()

    pointer = json.loads(current.read_text(encoding="utf-8"))
    failed = json.loads((tmp_path / pointer["report"]).read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["run_id"] == pointer["run_id"]
    marker, _, payload = capsys.readouterr().out.strip().partition(" ")
    assert marker == "BASELINE_SMOKE_OK"
    passed = json.loads(payload)
    assert passed["status"] == "success"
    assert passed["run_id"] != failed["run_id"]
    passed_report = tmp_path / "baseline-runs" / passed["run_id"] / "baseline-smoke.json"
    assert json.loads(passed_report.read_text(encoding="utf-8")) == passed


@pytest.mark.parametrize("flip_pixels", [True, False])
def test_transfer_flip_checks_pixels_as_well_as_boxes(flip_pixels: bool) -> None:
    source = (SCRIPTS / "colab_transfer_check.py").read_text(encoding="utf-8")
    start = source.index("box_xywh = ")
    end = source.index("\nrecord = ", start)

    class Wrapper:
        def __init__(self, p, transforms):
            self.transform = transforms
            self.contains_spatial = transforms[0]

        def __call__(self, labels):
            if self.contains_spatial:
                boxes = labels["instances"].bboxes
                boxes[:, 0] = 1 - boxes[:, 0]
                if flip_pixels:
                    labels["img"] = labels["img"][:, ::-1].copy()
            else:
                labels["img"] += 20
            return labels

    namespace = {
        "np": np,
        "A": SimpleNamespace(HorizontalFlip=lambda **kwargs: True, RandomBrightnessContrast=lambda **kwargs: False),
        "Instances": lambda boxes, **kwargs: SimpleNamespace(bboxes=boxes),
        "Albumentations": Wrapper,
    }
    expected = nullcontext() if flip_pixels else pytest.raises(AssertionError)
    with expected:
        exec(source[start:end], namespace)  # noqa: S102 - execute the checked-in synthetic fixture only
