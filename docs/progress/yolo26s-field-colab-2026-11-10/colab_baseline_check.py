"""Upload the sealed baseline package, verify it, and run a synthetic CPU smoke check."""

import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

EXPECTED_MANIFEST_SHA256 = "99ff187ab520c20cc047b9d69b5fe467a83896fd4546d08bf401d279e03670a3"
WORK = Path("/content/yolo26s-field")
BASELINE = WORK / "baseline"
DEPENDENCIES = WORK / "baseline-deps"
DRIVE = Path("/content/drive/MyDrive/TomatoAI/yolo26s-field-2026-11-10")

def verify_package(payload):
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("Baseline upload exceeds size limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError("Baseline package exceeds expected size")
        manifest_bytes = archive.read("upload-manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
            raise ValueError("Baseline sealed member manifest SHA-256 mismatch")
        manifest = json.loads(manifest_bytes)
        if len(archive.namelist()) != len(manifest) + 1 or set(archive.namelist()) != set(manifest) | {"upload-manifest.json"}:
            raise ValueError("Unexpected archive members")
        for name, digest in manifest.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe archive member")
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Baseline member mismatch: {name}")


def run_smoke(run_dir):
    from google.colab import files

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload only baseline-upload.zip")
    payload = next(iter(uploaded.values()))
    verify_package(payload)
    BASELINE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(BASELINE)

    # Keep the existing detection environment separate from baseline Pillow.
    if not (DEPENDENCIES / "PIL").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(DEPENDENCIES), "Pillow==13.0.0"], check=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(DEPENDENCIES), str(BASELINE)))
    environment["YOLO_CONFIG_DIR"] = str(WORK / "baseline-settings")
    verify = subprocess.run([sys.executable, "-m", "ai.artifacts", "verify", "--bundle", "ai/artifacts/yolo26s-cls-v1", "--release"], cwd=BASELINE, env=environment, text=True, capture_output=True, check=False)
    (run_dir / "baseline-verify.log").write_text(verify.stdout + verify.stderr, encoding="utf-8")
    if verify.returncode:
        raise RuntimeError(verify.stdout + verify.stderr)

    smoke = '''
    if not __debug__:
        raise RuntimeError("Baseline smoke requires assertions enabled")
    import json
    import sys
    from dataclasses import asdict
    from importlib.metadata import version
    from pathlib import Path
    import torch
    from PIL import Image
    from ai.inference import Predictor
    torch.set_num_threads(2)
    assert version("Pillow") == "13.0.0"
    predictor = Predictor(Path("ai/artifacts/yolo26s-cls-v1"))
    assert len(predictor.class_ids) == 8
    prediction = predictor.predict(Image.new("RGB", (321, 197), (60, 110, 45)))
    assert len(prediction.top_k) == 3
    record = {"versions": {name: version(name) for name in ("torch", "torchvision", "Pillow", "ultralytics")}, "class_ids": predictor.class_ids, "prediction": asdict(prediction)}
    Path(sys.argv[1]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    '''
    prediction = run_dir / "baseline-prediction.json"
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(smoke), str(prediction)], cwd=BASELINE, env=environment, text=True, capture_output=True, check=False)
    (run_dir / "baseline-smoke.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    evidence = json.loads(prediction.read_text(encoding="utf-8"))
    archive_copy = run_dir / "baseline-upload.zip"
    archive_copy.write_bytes(payload)
    archive_copy.replace(DRIVE / archive_copy.name)
    return evidence


def main():
    if not DRIVE.is_dir():
        raise RuntimeError("Expected mounted Drive task directory")
    run_id = uuid4().hex
    run_dir = DRIVE / "baseline-runs" / run_id
    run_dir.mkdir(parents=True)
    current = DRIVE / "baseline-smoke.json"
    report = run_dir / "baseline-smoke.json"
    if current.exists():
        previous = current.read_bytes()
        if "run_id" not in json.loads(previous):
            (run_dir / "previous-baseline-smoke.json").write_bytes(previous)

    status = {"run_id": run_id, "status": "running", "scope": "synthetic_smoke_only_not_accuracy"}
    report.write_text(json.dumps(status, indent=2), encoding="utf-8")
    # Publish the current attempt before upload. Older attempts can only update their own reports.
    pending = run_dir / "current.json"
    pending.write_text(json.dumps({"run_id": run_id, "report": report.relative_to(DRIVE).as_posix()}, indent=2), encoding="utf-8")
    pending.replace(current)
    try:
        if not __debug__:
            raise RuntimeError("Baseline smoke requires assertions enabled")
        status.update(run_smoke(run_dir), status="success")
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        pending.write_text(json.dumps(status, indent=2), encoding="utf-8")
        pending.replace(report)
    print("BASELINE_SMOKE_OK", json.dumps(status))


if __name__ == "__main__":
    main()
