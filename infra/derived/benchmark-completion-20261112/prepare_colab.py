"""Package exact frozen DINO weights, evaluation images and inference sources."""

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    target = OUT / "benchmark-inputs.zip"
    assert not target.exists()
    pins = {"purpose": "Frozen-checkpoint inference only", "files": {}, "datasets": {}}
    frozen = json.loads((ROOT / "infra/derived/dinov2-improvement-v1/results/frozen-models.json").read_text())
    with ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=1) as output:
        def add_bytes(name, data):
            assert name not in pins["files"]
            output.writestr(name, data)
            pins["files"][name] = hashlib.sha256(data).hexdigest()

        for name in ["ai/__init__.py", "ai/dinov2_field.py", "ai/dinov3_field.py", "ai/dinov2_improve.py"]:
            add_bytes(name, (ROOT / name).read_bytes())
        add_bytes("colab_predict.py", (OUT / "colab_predict.py").read_bytes())
        for name, archive in [
            ("prior", "infra/derived/dinov2-colab-v1/results/tomato-dinov2-results-20261111.zip"),
            ("candidate", "infra/derived/dinov2-improvement-v1/results/dinov2-improvement-model-20261111.zip"),
        ]:
            with ZipFile(ROOT / archive) as source:
                for member, expected in frozen["weights"][name].items():
                    dest = f"models/{name}/{member}"
                    digest = hashlib.sha256()
                    with source.open("model/" + member) as src, output.open(dest, "w") as dst:
                        while chunk := src.read(1024 * 1024):
                            digest.update(chunk)
                            dst.write(chunk)
                    assert digest.hexdigest() == expected
                    pins["files"][dest] = expected
            print("PACKED MODEL", name, flush=True)
        definitions = [
            ("historic", "dinov2-field-v2", "test.csv", "test", 2647),
            ("validation", "dinov2-field-v2", "train-val.csv", "val", 1642),
            ("taiwan", "dinov2-external-v1", "test.csv", "test", 460),
        ]
        for name, folder, manifest, split, count in definitions:
            base = ROOT / "infra/derived" / folder
            rows = [row for row in read(base / manifest) if row["split"] == split]
            assert len(rows) == count
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            add_bytes(f"data/{name}.csv", stream.getvalue().encode())
            pins["datasets"][name] = {"n": count, "original_manifest": str((base / manifest).relative_to(ROOT)), "original_manifest_sha256": hashlib.sha256((base / manifest).read_bytes()).hexdigest(), "split": split}
            for row in rows:
                path = "data/" + row["path"]
                image = (base / row["path"]).read_bytes()
                assert hashlib.sha256(image).hexdigest() == row["sha256"]
                if path in pins["files"]:
                    assert pins["files"][path] == row["sha256"]
                else:
                    add_bytes(path, image)
            print("PACKED DATA", name, len(rows), flush=True)
        for name, file in [("prior", "external-prior-size280.csv"), ("candidate", "external-candidate.csv")]:
            add_bytes(f"parity/{name}.csv", (ROOT / "infra/derived/dinov2-improvement-v1/results" / file).read_bytes())
        output.writestr("inputs.json", json.dumps(pins, indent=2))
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(pins["files"]) + 1
    with target.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {"archive": target.name, "bytes": target.stat().st_size, "sha256": sha, "files": len(pins["files"]), "datasets": pins["datasets"]}
    (OUT / "colab-input-receipt.json").write_text(json.dumps(receipt, indent=2))
    (OUT / "colab-inputs.json").write_text(json.dumps(pins, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
