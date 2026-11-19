"""Generate self-contained Colab notebook cell scripts for CNN+BiGRU training and holdout eval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_RUN_ID = "cnn-bigru-20261118"
DEFAULT_WORKDIR = "/content/cnn-bigru-run"
DEFAULT_EXPERIMENTS = ("G0", "G1", "G2", "G3")
_REPO_ROOT = Path(__file__).resolve().parents[3]

_PREFLIGHT_PROBE = """import importlib.metadata as meta, json, PIL, safetensors, torch, torchvision, numpy
assert meta.version("Pillow") == "13.0.0", meta.version("Pillow")
assert PIL.__version__ == "13.0.0", PIL.__version__
print(json.dumps({
    "pillow": PIL.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "safetensors": safetensors.__version__,
    "numpy": numpy.__version__,
}, sort_keys=True))
"""

_PREFLIGHT_BODY = """
run_id = __RUN_ID__
workdir = Path(__WORKDIR__)
workdir.mkdir(parents=True, exist_ok=True)
runtime_deps = workdir / "runtime-deps"
runtime_deps.mkdir(parents=True, exist_ok=True)
run_nonce = secrets.token_hex(8)
assert torch.cuda.is_available(), "Colab GPU runtime required"
disk_free = shutil.disk_usage("/content").free
assert disk_free >= 4 * 1024**3, f"Need >= 4 GiB free on /content, got {disk_free}"

pip_log = workdir / "setup-pip.log"
with pip_log.open("w", encoding="utf-8") as log:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "--no-input", "--no-deps",
            "--target", str(runtime_deps), "Pillow==13.0.0",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )

env = os.environ.copy()
env["PYTHONPATH"] = str(runtime_deps) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
probe = __PROBE__
versions = json.loads(subprocess.check_output([sys.executable, "-c", probe], env=env, text=True))
info = {
    "run_id": run_id,
    "run_nonce": run_nonce,
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version.split()[0],
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "disk_free_bytes": disk_free,
    "runtime_deps": str(runtime_deps),
    "packages": versions,
}
(workdir / "preflight.json").write_text(json.dumps(info, indent=2) + "\\n", encoding="utf-8")
print("CNN_GRU_PREFLIGHT", json.dumps(info, sort_keys=True), flush=True)
"""

_UPLOAD_BODY = """
MAX_PART_BYTES = 52_428_800
run_id = __RUN_ID__
target = Path(__TARGET_DIR__)
assert str(target) == "/content", f"Expected target_dir='/content', got {target}"
drive_dir = Path("/content/drive/MyDrive/TomatoAI") / run_id
archives = __ARCHIVES__
code_spec = __CODE_SPEC__


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_from_drive(name: str, expected_bytes: int | None, expected_sha: str | None) -> bool:
    dst = target / name
    if dst.is_file() and (expected_bytes is None or dst.stat().st_size == expected_bytes):
        if expected_sha is None or sha256_file(dst) == expected_sha:
            return True
    src = drive_dir / name
    if drive_dir.is_dir() and src.is_file():
        if expected_bytes is not None and src.stat().st_size != expected_bytes:
            return False
        if expected_sha is not None and sha256_file(src) != expected_sha:
            return False
        tmp = dst.with_name(dst.name + ".pending")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return True
    return False


for spec in archives:
    archive_name = spec["archive_name"]
    total_bytes = spec["total_bytes"]
    archive_sha = spec["sha256"]
    archive_path = target / archive_name
    if copy_from_drive(archive_name, total_bytes, archive_sha):
        print("ARCHIVE_READY", archive_name, total_bytes, archive_sha, flush=True)
        continue
    part_paths = []
    for part in spec["parts"]:
        pname = part["part_name"]
        pbytes = part["bytes"]
        psha = part["sha256"]
        assert pbytes <= MAX_PART_BYTES, f"{pname} exceeds 50 MiB limit: {pbytes}"
        ppath = target / pname
        if not copy_from_drive(pname, pbytes, psha):
            while not (ppath.is_file() and ppath.stat().st_size == pbytes and sha256_file(ppath) == psha):
                if ppath.exists():
                    ppath.unlink()
                print("UPLOAD_PART_WAIT", pname, pbytes, flush=True)
                _ = files.upload(target_dir="/content")
        assert ppath.stat().st_size == pbytes and sha256_file(ppath) == psha, pname
        part_paths.append(ppath)
    tmp_zip = archive_path.with_name(archive_name + ".pending")
    with tmp_zip.open("wb") as out:
        for ppath in part_paths:
            with ppath.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    os.replace(tmp_zip, archive_path)
    assert archive_path.stat().st_size == total_bytes and sha256_file(archive_path) == archive_sha, archive_name
    print("ARCHIVE_ASSEMBLED", archive_name, total_bytes, archive_sha, flush=True)

cname = code_spec["name"]
cbytes = code_spec.get("bytes")
csha = code_spec.get("sha256")
cpath = target / cname
if not copy_from_drive(cname, cbytes, csha):
    while not (
        cpath.is_file()
        and (cbytes is None or cpath.stat().st_size == cbytes)
        and (csha is None or sha256_file(cpath) == csha)
    ):
        if cpath.exists():
            cpath.unlink()
        print("UPLOAD_CODE_WAIT", cname, flush=True)
        _ = files.upload(target_dir="/content")
assert cpath.is_file()
if cbytes is not None:
    assert cpath.stat().st_size == cbytes, cname
if csha is not None:
    assert sha256_file(cpath) == csha, cname
print("UPLOAD_ALL_VERIFIED", cname, cpath.stat().st_size, sha256_file(cpath), flush=True)
"""

_FROZEN_INPUT_CHECK = """
def check_frozen_inputs(lock, experiment, registration_path, manifest_path):
    if (
        lock.get("experiment_id") != experiment
        or lock.get("registration_sha256") != sha256_file(registration_path)
        or lock.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError(f"Conflicting frozen experiment inputs: {experiment}")
"""

_TRAIN_BODY = """
CONTENT = Path("/content")
RUN_ID = __RUN_ID__
WORKDIR = Path(__WORKDIR__)
EXPERIMENTS = __EXPERIMENTS__
CNN_GRU_FILES = ("leaf-classifier.json", "model.safetensors", "cnn-gru-release.json")
MAX_WEIGHT_BYTES = 256_000_000


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_unzip(archive: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        entries = zf.infolist()
        assert len(entries) == len({e.filename for e in entries}), archive.name
        for entry in entries:
            assert not stat.S_ISLNK(entry.external_attr >> 16), entry.filename
            target = (dest / entry.filename).resolve()
            assert target.is_relative_to(dest_resolved), entry.filename
        zf.extractall(dest)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".pending")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


code_dir = WORKDIR / "code"
safe_unzip(CONTENT / "cnn-gru-code.zip", code_dir)
reg = json.loads((code_dir / "registration.json").read_text(encoding="utf-8"))
bundle_hashes = json.loads((code_dir / "bundle-hashes.json").read_text(encoding="utf-8"))
for rel, expected in bundle_hashes["files_sha256"].items():
    assert sha256_file(code_dir / rel) == expected, rel

data_dir = WORKDIR / "data"
if not (data_dir / ".extracted_ok").is_file():
    assert sha256_file(CONTENT / "dinov3-colab-input.zip") == reg["inputs_sha256"]["dinov3_colab_input_zip"]
    assert sha256_file(CONTENT / "dinov3-additions.zip") == reg["inputs_sha256"]["dinov3_additions_zip"]
    safe_unzip(CONTENT / "dinov3-colab-input.zip", WORKDIR / "raw-input")
    safe_unzip(CONTENT / "dinov3-additions.zip", WORKDIR / "raw-input")
    shutil.copytree(WORKDIR / "raw-input/data", data_dir, dirs_exist_ok=True)
    (data_dir / ".extracted_ok").write_text("ok\\n", encoding="utf-8")

manifest_dst = data_dir / "train-val-calib.csv"
atomic_copy(code_dir / "manifests/train-val-calib.csv", manifest_dst)
assert sha256_file(manifest_dst) == reg["inputs_sha256"]["three_way_manifest_csv"]

runtime_deps = WORKDIR / "runtime-deps"
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([str(runtime_deps), str(code_dir)]) + (
    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
)
subprocess.run(
    [sys.executable, "-c", "import importlib.metadata as m; assert m.version('Pillow') == '13.0.0'"],
    env=env,
    check=True,
)

drive_root = Path("/content/drive/MyDrive")
drive_run = drive_root / "TomatoAI" / RUN_ID
runs_dir = WORKDIR / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)
frozen_all = {"status": "frozen_before_holdout_prediction", "run_id": RUN_ID, "experiments": {}}

for exp_id in EXPERIMENTS:
    exp_cfg = reg["experiments"][exp_id]
    out_dir = runs_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_exp = drive_run / exp_id
    frozen_path = out_dir / "frozen-model.json"
    if drive_root.is_dir() and not frozen_path.is_file():
        if (drive_exp / "frozen-model.json").is_file():
            for item in drive_exp.iterdir():
                if item.is_file():
                    atomic_copy(item, out_dir / item.name)
            if (drive_exp / "model").is_dir():
                for sub in (drive_exp / "model").iterdir():
                    if sub.is_file():
                        atomic_copy(sub, out_dir / "model" / sub.name)
        elif (drive_exp / "last.pt").is_file() and not (out_dir / "last.pt").is_file():
            for item in drive_exp.iterdir():
                if item.is_file():
                    atomic_copy(item, out_dir / item.name)
    if frozen_path.is_file():
        frozen_exp = json.loads(frozen_path.read_text(encoding="utf-8"))
        check_frozen_inputs(frozen_exp, exp_id, code_dir / "registration.json", manifest_dst)
        model_dir = out_dir / "model"
        if (
            frozen_exp.get("status") == "frozen_before_holdout_prediction"
            and all(
                (model_dir / fname).is_file()
                and sha256_file(model_dir / fname) == frozen_exp.get("sha256", {}).get(fname)
                for fname in CNN_GRU_FILES
            )
        ):
            frozen_all["experiments"][exp_id] = {
                "frozen_model_sha256": sha256_file(frozen_path),
                **frozen_exp,
            }
            print("CNN_GRU_SKIP_FROZEN", exp_id, frozen_exp.get("selected_epoch"), frozen_exp.get("selected_variant"), flush=True)
            continue
    cmd = [
        sys.executable, "-u", "-m", "ai.cnn_gru_train",
        "--registration", str(code_dir / "registration.json"),
        "--experiment", exp_id,
        "--data", str(data_dir),
        "--manifest", str(manifest_dst),
        "--output", str(out_dir),
    ]
    if drive_root.is_dir():
        cmd.extend(["--drive-dir", str(drive_exp)])
    if (out_dir / "last.pt").is_file():
        cmd.append("--resume")
    with (out_dir / "train.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            cmd, cwd=code_dir, env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            rc = proc.wait()
            if rc:
                raise subprocess.CalledProcessError(rc, cmd)
    model_dir = out_dir / "model"
    found_files = tuple(sorted(p.name for p in model_dir.iterdir() if p.is_file()))
    assert found_files == tuple(sorted(CNN_GRU_FILES)), found_files
    weights_path = model_dir / "model.safetensors"
    assert 0 < weights_path.stat().st_size <= MAX_WEIGHT_BYTES
    leaf_meta = json.loads((model_dir / "leaf-classifier.json").read_text(encoding="utf-8"))
    selected_epoch = leaf_meta["selected_epoch"]
    selected_variant = leaf_meta["selected_variant"]
    assert (out_dir / f"best-epoch-{selected_epoch}-{selected_variant}.pt").is_file()
    assert (out_dir / "last.pt").is_file()
    frozen_exp = {
        "status": "frozen_before_holdout_prediction",
        "experiment_id": exp_id,
        "name": exp_cfg["name"],
        "selected_epoch": selected_epoch,
        "selected_variant": selected_variant,
        "guardrail_passed": leaf_meta["guardrail_passed"],
        "sha256": {name: sha256_file(model_dir / name) for name in CNN_GRU_FILES},
        "registration_sha256": sha256_file(code_dir / "registration.json"),
        "manifest_sha256": sha256_file(manifest_dst),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    frozen_path = out_dir / "frozen-model.json"
    tmp_f = frozen_path.with_suffix(".pending")
    tmp_f.write_text(json.dumps(frozen_exp, indent=2) + "\\n", encoding="utf-8")
    os.replace(tmp_f, frozen_path)
    frozen_all["experiments"][exp_id] = {
        "frozen_model_sha256": sha256_file(frozen_path),
        **frozen_exp,
    }
    if drive_root.is_dir():
        for fname in CNN_GRU_FILES:
            atomic_copy(model_dir / fname, drive_exp / "model" / fname)
        for name in ("frozen-model.json", "history.json", "calibration.json", "train.log"):
            if (out_dir / name).is_file():
                atomic_copy(out_dir / name, drive_exp / name)

frozen_root = WORKDIR / "frozen-model.json"
tmp_root = frozen_root.with_suffix(".pending")
tmp_root.write_text(json.dumps(frozen_all, indent=2) + "\\n", encoding="utf-8")
os.replace(tmp_root, frozen_root)

train_zip = CONTENT / "cnn-bigru-train-results.zip"
with ZipFile(train_zip.with_suffix(".pending"), "w", ZIP_STORED) as zf:
    zf.write(frozen_root, "frozen-model.json")
    for exp_id in EXPERIMENTS:
        out_dir = runs_dir / exp_id
        for name in CNN_GRU_FILES:
            zf.write(out_dir / "model" / name, f"{exp_id}/model/{name}")
        for name in ("frozen-model.json", "history.json", "calibration.json", "train.log"):
            if (out_dir / name).is_file():
                zf.write(out_dir / name, f"{exp_id}/{name}")
os.replace(train_zip.with_suffix(".pending"), train_zip)
if drive_root.is_dir():
    atomic_copy(train_zip, drive_run / train_zip.name)
print("CNN_GRU_PHASE1_FROZEN", train_zip.name, sha256_file(train_zip), flush=True)
"""

_HOLDOUT_EVAL_SCRIPT = """import csv, hashlib, importlib.metadata as meta, json, sys
from pathlib import Path
from PIL import Image
import torch
from ai.cnn_gru_inference import CnnGruPredictor, validate_cnn_gru_export
from ai.cnn_gru_model import cnn_gru_eval_transform
from ai.dinov2_field import summarize
from ai.dinov3_field import CLASSES
from ai.dinov3_inference import prediction_from_logits

assert meta.version("Pillow") == "13.0.0", meta.version("Pillow")
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

workdir = Path(sys.argv[1])
experiments = sys.argv[2].split(",")
holdout = workdir / "holdout"
datasets = {}
for name in ("legacy", "external", "slif"):
    root = holdout / name
    rows = list(csv.DictReader((root / "test.csv").open(encoding="utf-8", newline="")))
    assert {r["split"] for r in rows} == {"test"}
    for r in rows:
        with (root / r["path"]).open("rb") as s:
            assert hashlib.file_digest(s, "sha256").hexdigest() == r["sha256"], r["path"]
    datasets[name] = (root, rows)

parity_dir = workdir / "parity-png"
parity_dir.mkdir(parents=True, exist_ok=True)
pick_specs = [
    ("external", datasets["external"][1][0]),
    ("external", datasets["external"][1][-1]),
    ("legacy", datasets["legacy"][1][0]),
    ("legacy", datasets["legacy"][1][-1]),
    ("slif", datasets["slif"][1][0]),
]
parity_inputs = []
for idx, (ds_name, row) in enumerate(pick_specs):
    src = holdout / ds_name / row["path"]
    png_path = parity_dir / f"parity-{idx:02d}-{ds_name}.png"
    with Image.open(src) as im:
        im.convert("RGB").save(png_path, format="PNG", compress_level=0)
    with png_path.open("rb") as s:
        png_sha = hashlib.file_digest(s, "sha256").hexdigest()
    parity_inputs.append({
        "index": idx, "dataset": ds_name, "source_path": row["path"],
        "class_id": row["class_id"], "png_name": png_path.name, "png_sha256": png_sha,
        "png_path": str(png_path),
    })

device = torch.device("cuda")
for exp_id in experiments:
    exp_dir = workdir / "runs" / exp_id
    model_dir = exp_dir / "model"
    meta_cfg = validate_cnn_gru_export(model_dir, require_trained=True)
    leaf_cfg = json.loads((model_dir / "leaf-classifier.json").read_text(encoding="utf-8"))
    size = int(meta_cfg.get("size") or leaf_cfg.get("size") or 288)
    prep = (
        meta_cfg.get("preprocessing")
        or leaf_cfg.get("preprocessing")
        or {
            "mean": meta_cfg.get("mean") or leaf_cfg["mean"],
            "std": meta_cfg.get("std") or leaf_cfg["std"],
        }
    )
    temp = float(meta_cfg["temperature"])
    thresh = float(meta_cfg["threshold"])
    ver = str(meta_cfg["model_version"])
    cpu_pred = CnnGruPredictor(model_dir)
    gpu_pred = CnnGruPredictor(model_dir)
    gpu_model = gpu_pred._model.to(device).eval()
    transform = cnn_gru_eval_transform(size, prep["mean"], prep["std"])

    holdout_metrics = {}
    for ds_name, (root, rows) in datasets.items():
        records, labels = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                batch_rows = rows[start : start + 32]
                tensors = []
                for r in batch_rows:
                    with Image.open(root / r["path"]) as im:
                        im.load()
                        tensors.append(transform(im))
                x = torch.stack(tensors).to(device=device, dtype=torch.float32)
                logits = 0.5 * (gpu_model(x).float() + gpu_model(torch.flip(x, dims=[-1])).float())
                for r, row_logits in zip(batch_rows, logits.cpu(), strict=True):
                    pred = prediction_from_logits(row_logits.unsqueeze(0), ver, temperature=temp, threshold=thresh)
                    top = pred.top_k[0].class_id if pred.top_k else CLASSES[int(row_logits.argmax().item())]
                    conf = pred.top_k[0].score if pred.top_k else 0.0
                    raw_top = CLASSES[int(row_logits.argmax().item())]
                    labels.append(raw_top)
                    records.append({
                        **r, "prediction": pred.class_id or "uncertain",
                        "raw_top1": raw_top, "top1_positive": top, "confidence": conf, "status": pred.status,
                        "logits": json.dumps([round(float(v), 7) for v in row_logits.tolist()]),
                    })
        pred_csv = exp_dir / f"{exp_id}-{ds_name}-predictions.csv"
        with pred_csv.open("w", encoding="utf-8", newline="") as stream:
            fields = ("path", "sha256", "class_id", "source", "domain", "group",
                      "prediction", "raw_top1", "top1_positive", "confidence", "status", "logits")
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        holdout_metrics[ds_name] = summarize(rows, labels)
    (exp_dir / "holdout-metrics.json").write_text(json.dumps(holdout_metrics, indent=2) + "\\n", encoding="utf-8")

    parity_rows = []
    max_abs = 0.0
    for item in parity_inputs:
        with Image.open(item["png_path"]) as im:
            im.load()
            cpu_out = cpu_pred.predict(im)
            x_cpu = transform(im).unsqueeze(0).to(dtype=torch.float32)
            x_gpu = x_cpu.to(device=device)
            with torch.inference_mode():
                gpu_logits = 0.5 * (gpu_model(x_gpu).float() + gpu_model(torch.flip(x_gpu, dims=[-1])).float())
                cpu_logits = 0.5 * (cpu_pred._model(x_cpu).float() + cpu_pred._model(torch.flip(x_cpu, dims=[-1])).float())
            diff = float((gpu_logits.cpu() - cpu_logits).abs().max().item())
            max_abs = max(max_abs, diff)
            assert diff < 1e-5, f"Parity diff {diff} >= 1e-5 on {item['png_name']}"
            gpu_pred_out = prediction_from_logits(gpu_logits.cpu(), ver, temperature=temp, threshold=thresh)
            assert gpu_pred_out.class_id == cpu_out.class_id and gpu_pred_out.status == cpu_out.status
            parity_rows.append({
                "index": item["index"], "dataset": item["dataset"], "png_name": item["png_name"],
                "png_sha256": item["png_sha256"], "source_path": item["source_path"],
                "true_class_id": item["class_id"],
                "cpu_status": cpu_out.status, "cpu_class_id": cpu_out.class_id,
                "cpu_logits_fp32": cpu_logits.squeeze(0).tolist(),
                "gpu_logits_fp32": gpu_logits.squeeze(0).cpu().tolist(),
                "max_abs_diff_gpu_vs_cpu": diff,
            })
    (exp_dir / "parity-png-fp32.json").write_text(
        json.dumps({"max_abs_diff": max_abs, "samples": parity_rows}, indent=2) + "\\n",
        encoding="utf-8",
    )
    del cpu_pred, gpu_pred, gpu_model
    torch.cuda.empty_cache()
    print("HOLDOUT_AND_PARITY_OK", exp_id, "max_abs_diff", max_abs, flush=True)
"""

_HOLDOUT_BODY = """
CONTENT = Path("/content")
RUN_ID = __RUN_ID__
WORKDIR = Path(__WORKDIR__)
EXPERIMENTS = __EXPERIMENTS__


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_unzip(archive: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        entries = zf.infolist()
        assert len(entries) == len({e.filename for e in entries}), archive.name
        for entry in entries:
            assert not stat.S_ISLNK(entry.external_attr >> 16), entry.filename
            target = (dest / entry.filename).resolve()
            assert target.is_relative_to(dest_resolved), entry.filename
        zf.extractall(dest)


code_dir = WORKDIR / "code"
reg = json.loads((code_dir / "registration.json").read_text(encoding="utf-8"))
frozen_root = json.loads((WORKDIR / "frozen-model.json").read_text(encoding="utf-8"))
assert frozen_root["status"] == "frozen_before_holdout_prediction"
for exp_id in EXPERIMENTS:
    f_path = WORKDIR / "runs" / exp_id / "frozen-model.json"
    assert sha256_file(f_path) == frozen_root["experiments"][exp_id]["frozen_model_sha256"], exp_id
    f_exp = json.loads(f_path.read_text(encoding="utf-8"))
    check_frozen_inputs(f_exp, exp_id, code_dir / "registration.json", code_dir / "manifests/train-val-calib.csv")
    assert f_exp["status"] == "frozen_before_holdout_prediction", exp_id
    for fname, expected in f_exp["sha256"].items():
        assert sha256_file(WORKDIR / "runs" / exp_id / "model" / fname) == expected, (exp_id, fname)

holdout_dir = WORKDIR / "holdout"
assert sha256_file(CONTENT / "dinov3-colab-holdout.zip") == reg["inputs_sha256"]["dinov3_colab_holdout_zip"]
assert sha256_file(CONTENT / "dinov3-slif-pilot.zip") == reg["inputs_sha256"]["dinov3_slif_pilot_zip"]
safe_unzip(CONTENT / "dinov3-colab-holdout.zip", holdout_dir)
safe_unzip(CONTENT / "dinov3-slif-pilot.zip", holdout_dir)
assert sha256_file(holdout_dir / "legacy/test.csv") == reg["inputs_sha256"]["legacy_test_csv"]
assert sha256_file(holdout_dir / "external/test.csv") == reg["inputs_sha256"]["external_test_csv"]
assert sha256_file(holdout_dir / "slif/test.csv") == reg["inputs_sha256"]["slif_test_csv"]

eval_script = __EVAL_SCRIPT__
eval_path = WORKDIR / "run_holdout_eval.py"
eval_path.write_text(eval_script, encoding="utf-8")
runtime_deps = WORKDIR / "runtime-deps"
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([str(runtime_deps), str(code_dir)]) + (
    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
)
subprocess.run([sys.executable, "-u", str(eval_path), str(WORKDIR), ",".join(EXPERIMENTS)], env=env, check=True)

test_zip = CONTENT / "cnn-bigru-test-results.zip"
with ZipFile(test_zip.with_suffix(".pending"), "w", ZIP_STORED) as zf:
    zf.write(WORKDIR / "frozen-model.json", "frozen-model.json")
    for png in sorted((WORKDIR / "parity-png").glob("*.png")):
        zf.write(png, f"parity-png/{png.name}")
    for exp_id in EXPERIMENTS:
        exp_dir = WORKDIR / "runs" / exp_id
        for item in sorted(exp_dir.glob("*.csv")) + sorted(exp_dir.glob("*.json")):
            zf.write(item, f"{exp_id}/{item.name}")
os.replace(test_zip.with_suffix(".pending"), test_zip)
drive_root = Path("/content/drive/MyDrive")
if drive_root.is_dir():
    drive_run = drive_root / "TomatoAI" / RUN_ID
    drive_run.mkdir(parents=True, exist_ok=True)
    tmp_drive = drive_run / (test_zip.name + ".pending")
    shutil.copy2(test_zip, tmp_drive)
    os.replace(tmp_drive, drive_run / test_zip.name)
print("CNN_GRU_PHASE2_DONE", test_zip.name, sha256_file(test_zip), flush=True)
"""


def _render(header: str, body: str, replacements: dict[str, str]) -> str:
    text = header + body.lstrip("\n")
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def preflight_cell(
    run_id: str = DEFAULT_RUN_ID,
    workdir: str = DEFAULT_WORKDIR,
) -> str:
    """Return Colab preflight cell checking GPU, disk, and isolated Pillow==13.0.0."""
    header = (
        'if not __debug__: raise RuntimeError("Assertions required")\n'
        "import importlib.metadata as meta\n"
        "import json\nimport os\nimport secrets\nimport shutil\n"
        "import subprocess\nimport sys\n"
        "from datetime import datetime, timezone\nfrom pathlib import Path\n"
        "import torch\n\n"
    )
    return _render(
        header,
        _PREFLIGHT_BODY,
        {
            "__RUN_ID__": repr(run_id),
            "__WORKDIR__": repr(workdir),
            "__PROBE__": repr(_PREFLIGHT_PROBE),
        },
    )


def upload_cell(
    manifest_parts: dict[str, Any] | None = None,
    *,
    code_bundle: dict[str, Any] | None = None,
    run_id: str = DEFAULT_RUN_ID,
    target_dir: str = "/content",
) -> str:
    """Return Colab upload cell supporting Drive staging and <=50 MiB chunked files.upload."""
    bundle_dir = _REPO_ROOT / ".scratch" / run_id / "colab-bundle"
    if manifest_parts is None:
        mp_path = bundle_dir / "parts/manifest-parts.json"
        if mp_path.is_file():
            manifest_parts = json.loads(mp_path.read_text(encoding="utf-8"))
    if code_bundle is None:
        bh_path = bundle_dir / "bundle-hashes.json"
        if bh_path.is_file():
            bh = json.loads(bh_path.read_text(encoding="utf-8"))
            if isinstance(bh.get("code_zip"), dict):
                code_bundle = bh["code_zip"]
    archives = (manifest_parts or {}).get("archives", [])
    code_spec = code_bundle or {
        "name": "cnn-gru-code.zip",
        "bytes": None,
        "sha256": None,
    }
    header = (
        'if not __debug__: raise RuntimeError("Assertions required")\n'
        "import hashlib\nimport json\nimport os\nimport shutil\n"
        "from pathlib import Path\nfrom google.colab import files\n\n"
    )
    return _render(
        header,
        _UPLOAD_BODY,
        {
            "__RUN_ID__": repr(run_id),
            "__TARGET_DIR__": repr(target_dir),
            "__ARCHIVES__": repr(archives),
            "__CODE_SPEC__": repr(code_spec),
        },
    )


def train_cell(
    experiments: tuple[str, ...] = DEFAULT_EXPERIMENTS,
    *,
    run_id: str = DEFAULT_RUN_ID,
    workdir: str = DEFAULT_WORKDIR,
) -> str:
    """Return Colab Phase 1 training + calibration + frozen-model lock cell."""
    header = (
        'if not __debug__: raise RuntimeError("Assertions required")\n'
        "import hashlib\nimport json\nimport os\nimport shutil\nimport stat\n"
        "import subprocess\nimport sys\n"
        "from datetime import datetime, timezone\nfrom pathlib import Path\n"
        "from zipfile import ZIP_STORED, ZipFile\n\n"
    )
    return _render(
        header,
        _FROZEN_INPUT_CHECK + _TRAIN_BODY,
        {
            "__RUN_ID__": repr(run_id),
            "__WORKDIR__": repr(workdir),
            "__EXPERIMENTS__": repr(tuple(experiments)),
        },
    )


def holdout_eval_cell(
    experiments: tuple[str, ...] = DEFAULT_EXPERIMENTS,
    *,
    run_id: str = DEFAULT_RUN_ID,
    workdir: str = DEFAULT_WORKDIR,
) -> str:
    """Return Colab Phase 2 holdout evaluation + 5 lossless PNG FP32 parity cell."""
    header = (
        'if not __debug__: raise RuntimeError("Assertions required")\n'
        "import hashlib\nimport json\nimport os\nimport shutil\nimport stat\n"
        "import subprocess\nimport sys\n"
        "from pathlib import Path\n"
        "from zipfile import ZIP_STORED, ZipFile\n\n"
    )
    return _render(
        header,
        _FROZEN_INPUT_CHECK + _HOLDOUT_BODY,
        {
            "__RUN_ID__": repr(run_id),
            "__WORKDIR__": repr(workdir),
            "__EXPERIMENTS__": repr(tuple(experiments)),
            "__EVAL_SCRIPT__": repr(_HOLDOUT_EVAL_SCRIPT),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        choices=("preflight", "upload", "train", "holdout"),
        required=True,
    )
    args = parser.parse_args()
    builders: dict[str, Callable[[], str]] = {
        "preflight": preflight_cell,
        "upload": upload_cell,
        "train": train_cell,
        "holdout": holdout_eval_cell,
    }
    print(builders[args.cell]())


if __name__ == "__main__":
    main()
