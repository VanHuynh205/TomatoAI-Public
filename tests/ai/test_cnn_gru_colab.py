"""A Colab rerun may reuse only the experiment and inputs recorded in its lock."""

import hashlib
import json
import runpy
import subprocess
from importlib import import_module
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize("phase", ["train_cell", "holdout_eval_cell"])
@pytest.mark.parametrize(
    "stale", ["experiment_id", "registration_sha256", "manifest_sha256"]
)
def test_stale_frozen_experiment_stops_before_training_or_holdout(
    tmp_path: Path, phase: str, stale: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = import_module("docs.progress.2026-11-18-cnn-bigru.colab_cells")
    code = tmp_path / "code"
    (code / "manifests").mkdir(parents=True)
    manifest = b"synthetic manifest"
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    (code / "manifests/train-val-calib.csv").write_bytes(manifest)
    registration = json.dumps(
        {
            "inputs_sha256": {"three_way_manifest_csv": manifest_hash},
            "experiments": {"G0": {"name": "baseline"}},
        }
    ).encode()
    (code / "registration.json").write_bytes(registration)
    (code / "bundle-hashes.json").write_text('{"files_sha256": {}}')
    with ZipFile(tmp_path / "cnn-gru-code.zip", "w") as bundle:
        for path in code.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(code))
    data = tmp_path / "data"
    data.mkdir()
    (data / ".extracted_ok").touch()
    (data / "train-val-calib.csv").write_bytes(manifest)

    run = tmp_path / "runs/G0"
    model = run / "model"
    model.mkdir(parents=True)
    hashes = {}
    for name in ("leaf-classifier.json", "model.safetensors", "cnn-gru-release.json"):
        (model / name).write_bytes(b"fixture")
        hashes[name] = hashlib.sha256(b"fixture").hexdigest()
    lock = {
        "status": "frozen_before_holdout_prediction",
        "experiment_id": "G0",
        "registration_sha256": hashlib.sha256(registration).hexdigest(),
        "manifest_sha256": manifest_hash,
        "sha256": hashes,
        stale: "stale",
    }
    raw_lock = json.dumps(lock).encode()
    (run / "frozen-model.json").write_bytes(raw_lock)
    (tmp_path / "frozen-model.json").write_text(
        json.dumps(
            {
                "status": lock["status"],
                "run_id": cells.DEFAULT_RUN_ID,
                "experiments": {
                    "G0": {"frozen_model_sha256": hashlib.sha256(raw_lock).hexdigest()}
                },
            }
        )
    )
    script = getattr(cells, phase)(("G0",), workdir=str(tmp_path))
    script = script.replace('Path("/content")', f"Path({str(tmp_path)!r})")
    script = script.replace(
        'Path("/content/drive/MyDrive")', f"Path({str(tmp_path / 'drive')!r})"
    )
    # The Colab image's Pillow version probe is outside this regression.
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    cell_path = tmp_path / "cell.py"
    cell_path.write_text(script, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen experiment inputs"):
        runpy.run_path(str(cell_path))
    assert (run / "frozen-model.json").read_bytes() == raw_lock
