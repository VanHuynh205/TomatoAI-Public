"""Freeze retained candidate checkpoints and reuse the verified benchmark inputs."""

import hashlib
import io
import json
import sys
from importlib import import_module
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    prior = ROOT / "infra/derived/benchmark-completion-20261112"
    archive = OUT / "model-selection-inputs-v2.zip"
    registration = {
        "purpose": "Retrospective frozen-checkpoint comparison; no training",
        "selection": {"dataset": "taiwan", "slice": "all", "metric": "macro_f1", "tie_break": "field_tomato macro_f1, then smaller checkpoint"},
        "limits": "Taiwan/TOM2024 has three tomato classes plus negative controls. Previously inspected; selection makes it a development benchmark, not a fresh final test.",
        "files": {}, "models": [], "datasets": {"taiwan": 460, "historic": 2647, "validation": 1642},
        "unavailable": [f"YOLO26s-cls P03 {name}: head weights not saved" for name in ["ce-25", "kd-25", "ce-40", "kd-40", "ce-55", "kd-55"]],
    }
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=1) as output:
        def add(name, data):
            assert name not in registration["files"]
            output.writestr(name, data)
            registration["files"][name] = hashlib.sha256(data).hexdigest()

        with ZipFile(prior / "benchmark-inputs.zip") as source:
            pins = json.loads(source.read("inputs.json"))
            for name, expected in pins["files"].items():
                if name.startswith("data/"):
                    data = source.read(name)
                    assert hashlib.sha256(data).hexdigest() == expected
                    add(name, data)
        for path in (ROOT / "ai").glob("*.py"):
            add("ai/" + path.name, path.read_bytes())
        add("score.py", (OUT / "score.py").read_bytes())
        for number in range(1, 11):
            run = f"E{number:02}"
            folder = ROOT / "ai/runs" / run
            metadata = json.loads((folder / "metadata.json").read_text())
            assert digest(folder / "best.pth") == metadata["checkpoint_hashes"]["best.pth"]
            config = metadata["config"]
            family = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50", "yolo26s_cls": "YOLO26s-cls"}[config["architecture"]]
            label = family + " " + run + (" frozen" if number in (1, 3, 5, 7, 9) else " adapt" if number in (6, 8) else "")
            name = f"models/{run}/best.pth"
            add(name, (folder / "best.pth").read_bytes())
            add(f"models/{run}/metadata.json", (folder / "metadata.json").read_bytes())
            registration["models"].append({"id": run, "label": label, "family": family, "kind": "run", "path": name, "source": str(folder.relative_to(ROOT)).replace("\\", "/"), "epoch": metadata["best_epoch"], "datasets": ["taiwan", "historic", "validation"] if run in ("E02", "E04") else ["taiwan"]})
        base = ROOT / "docs/progress/yolo-image-recovery"
        decision = json.loads((base / "P04/decision.json").read_text())
        for epoch in (2, 4):
            path = base / f"P04/epoch-{epoch}.pth"
            assert digest(path) == decision["variants"][f"epoch-{epoch}"]["checkpoint_sha256"]
            name = f"models/P04/epoch-{epoch}.pth"
            add(name, path.read_bytes())
            registration["models"].append({"id": f"P04-e{epoch}", "label": f"YOLO26s-cls P04 epoch-{epoch}", "family": "YOLO26s-cls", "kind": "state", "path": name, "epoch": epoch, "source": str(path.relative_to(ROOT)).replace("\\", "/"), "datasets": ["taiwan"]})
        import torch
        from ai.train import _load_checkpoint

        torch.set_num_threads(2)
        interpolation = import_module("docs.progress.yolo-image-recovery.weight_interpolation")
        lock = json.loads((base / "p05-lock.json").read_text())
        input_names = ["ai/runs/E10/best.pth", "docs/progress/yolo-image-recovery/weight_interpolation.py"]
        input_names += [f"docs/progress/yolo-image-recovery/P04/epoch-{epoch}.pth" for epoch in (2, 4)]
        for relative in input_names:
            assert digest(ROOT / relative) == lock["files_sha256"][relative], relative
        assert list(interpolation.MASSES) == lock["masses"]
        add("provenance/p05-lock.json", (base / "p05-lock.json").read_bytes())
        add("provenance/weight_interpolation.py", (base / "weight_interpolation.py").read_bytes())
        original = _load_checkpoint(ROOT / "ai/runs/E10/best.pth", "best")["model"]
        for epoch in (2, 4):
            adapted = torch.load(base / f"P04/epoch-{epoch}.pth", map_location="cpu", weights_only=True)
            for mass in interpolation.MASSES:
                ident = f"P05-e{epoch}-{int(mass * 100)}"
                state = interpolation.interpolate(original, adapted, mass)
                buffer = io.BytesIO()
                torch.save(state, buffer)
                restored = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=True)
                assert all(torch.equal(value, restored[key]) for key, value in state.items())
                name = f"models/P05/{ident}.pth"
                add(name, buffer.getvalue())
                registration["models"].append({"id": ident, "label": f"YOLO26s-cls P05 e{epoch}-{int(mass * 100)}", "family": "YOLO26s-cls", "kind": "state", "path": name, "epoch": epoch, "source": "docs/progress/yolo-image-recovery/P05", "reconstruction": {"base": "E10", "adapted": f"P04-e{epoch}", "mass": mass, "function": "provenance/weight_interpolation.py:interpolate"}, "datasets": ["taiwan"]})
        for version in ("v1", "v2"):
            source = ROOT / f"infra/derived/dinov3-fast-track-v1/colab-v3-20261112/sampler-{version}/model"
            branch = "A" if version == "v1" else "B"
            dest = f"models/dinov3-v3-{branch}"
            for path in source.iterdir():
                if path.is_file():
                    add(dest + "/" + path.name, path.read_bytes())
            metadata = json.loads((source / "leaf-classifier.json").read_text())
            registration["models"].append({"id": f"dinov3-v3-{branch}", "label": f"DINOv3 v3 {branch}", "family": "DINOv3", "kind": "dinov3", "path": dest, "source": str(source.relative_to(ROOT)).replace("\\", "/"), "epoch": metadata["selected_epoch"], "datasets": ["taiwan", "historic", "validation"]})
        # Existing predictor output provides a regression reference on the same images.
        for model, name in [("E06", "b0-e06"), ("E08", "resnet-e08")]:
            add(f"parity/{model}.csv", (prior / f"{name}-taiwan-predictions.csv").read_bytes())
        output.writestr("registration.json", json.dumps(registration, ensure_ascii=False, indent=2))
    assert len(registration["models"]) == 18
    with ZipFile(archive) as check:
        assert check.testzip() is None
        assert len(check.namelist()) == len(registration["files"]) + 1
    (OUT / "registration-v2.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive), "models": len(registration["models"]), "runs": sum(len(m["datasets"]) for m in registration["models"])}
    (OUT / "input-receipt-v2.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
