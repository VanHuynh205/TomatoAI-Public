"""Package only baseline inference files and exclusion metadata for private Colab."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILES = [
    *[f"ai/{name}.py" for name in ("__init__", "inference", "artifacts", "contracts", "models", "preprocessing", "data", "yolo")],
    "ai/artifacts/yolo26s-cls-v1/metadata.json",
    "ai/artifacts/yolo26s-cls-v1/model.pth",
    "docs/progress/yolo26s-field-colab-2026-11-10/protected-splits.json",
]

def package_bytes(contents):
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in {**contents, "upload-manifest.json": json.dumps(hashes, indent=2).encode()}.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 10, 0, 0, 0))
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue(), hashes


if __name__ == "__main__":
    payload, hashes = package_bytes({name: (ROOT / name).read_bytes() for name in FILES})
    if hashes["ai/artifacts/yolo26s-cls-v1/model.pth"] != "322ec3442a5fa54e913da155e8964205ea964117a63fdeec0780820c30f6e56f":
        raise ValueError("Baseline weights checksum mismatch")
    output = HERE / "baseline-upload.zip"
    with output.open("xb") as stream:
        stream.write(payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    record = {"path": str(output.relative_to(ROOT)).replace("\\", "/"), "sha256": digest, "bytes": output.stat().st_size, "files_sha256": hashes}
    (HERE / "baseline-upload-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "files_sha256"}, indent=2))
