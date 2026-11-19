"""Label scratch artifacts by their verified role; never rename or delete them."""

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).parent


def classify(path):
    parts = path.parts
    name = path.name
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return "DISPOSABLE", "Cache có thể tạo lại; không phải dữ liệu nghiên cứu."
    if name == "_update-notifier-last-checked":
        return (
            "DISPOSABLE",
            "Marker rỗng của update notifier; không có kết quả nghiên cứu.",
        )
    group = parts[1]
    if group == "retrain-20261107":
        purpose = parts[2]
        if purpose.startswith("profile-"):
            if name in {"metadata.json", "performance.json"}:
                return (
                    "KEEP_EVIDENCE",
                    "Bằng chứng VRAM, optimizer update và profile trước train.",
                )
            if name in {"best.pth", "last.pth"}:
                return (
                    "DISPOSABLE",
                    "Checkpoint profile vài minibatch; không được dùng làm candidate.",
                )
        elif purpose.endswith("-temp"):
            return (
                "DISPOSABLE",
                "Fixture do pytest tạo; kết quả kiểm tra nằm trong docs/progress.",
            )
        return "REVIEW_REQUIRED", "Artifact retrain chưa khớp mục đích đã xác minh."
    if group.startswith("plantdoc-"):
        if path.suffix == ".py":
            return (
                "KEEP_REUSE",
                "Script audit/split; giữ nguyên hash, chỉ chạy lại theo lock và review gate.",
            )
        if path.suffix == ".csv" or name.endswith(".sha256"):
            return (
                "KEEP_REUSE",
                "Membership, baseline hoặc provenance cần cho đánh giá sau; không tự thêm vào train.",
            )
        return (
            "KEEP_EVIDENCE",
            "Bằng chứng duyệt nhãn, contact sheet hoặc kiểm chứng PlantDoc.",
        )
    if "cache" in group or group in {
        "d4-temp",
        "package-acl-1jzkixcm",
        "m6-runtime-temp",
        "m6-offline",
        "d5-damaged",
    }:
        return "DISPOSABLE", "Cache/fixture cũ có thể tái tạo; không xóa tự động."
    if path.suffix in {".py", ".ps1", ".spec", ".mjs"}:
        return (
            "KEEP_REUSE",
            "Script kiểm chứng hoặc thao tác cũ; đọc lại phạm vi và gate trước khi chạy.",
        )
    if path.suffix in {".json", ".jsonl", ".xml", ".log", ".md", ".txt", ".csv"}:
        return (
            "KEEP_EVIDENCE",
            "Log, cấu hình hoặc kết quả kiểm chứng lịch sử; giữ truy vết.",
        )
    if group in {"d3", "d5-inflight", "m45-cuda"}:
        return (
            "DISPOSABLE",
            "Fixture/profile cũ, không phải checkpoint candidate hoặc dữ liệu raw.",
        )
    return (
        "REVIEW_REQUIRED",
        "Chưa đủ căn cứ xác định mục đích; giữ nguyên và kiểm tra thủ công.",
    )


def main():
    assert (
        classify(Path(".scratch/retrain-20261107/profile-E06/best.pth"))[0]
        == "DISPOSABLE"
    )
    assert classify(Path(".scratch/plantdoc-split-v1/freeze.py"))[0] == "KEEP_REUSE"
    assert (
        classify(Path(".scratch/retrain-20261107/unrecognized/data.bin"))[0]
        == "REVIEW_REQUIRED"
    )
    before = json.loads((REPORT / "scratch-before.json").read_text(encoding="utf-8"))
    before = {key.replace("\\", "/"): value for key, value in before.items()}
    records = []
    for file in sorted((ROOT / ".scratch").rglob("*")):
        if not file.is_file() and not file.is_symlink():
            continue
        relative = file.relative_to(ROOT)
        stat = file.lstat()
        prior = before.get(relative.as_posix())
        change = (
            "new"
            if prior is None
            else "changed"
            if prior != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            else "preexisting_unchanged"
        )
        label, reason = classify(relative)
        if file.is_symlink():
            label, reason = (
                "REVIEW_REQUIRED",
                "Symlink; không đọc target ngoài phạm vi inventory.",
            )
        digest = ""
        if label in {"KEEP_REUSE", "KEEP_EVIDENCE"}:
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            {
                "path": relative.as_posix(),
                "change_since_snapshot": change,
                "label": label,
                "reason": reason,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256_retained_evidence": digest,
            }
        )
    assert len(records) == len({row["path"] for row in records})
    labels = Counter(row["label"] for row in records)
    changes = [
        row
        for row in records
        if row["change_since_snapshot"] != "preexisting_unchanged"
    ]
    missing = sorted(set(before) - {row["path"] for row in records})
    with (REPORT / "scratch-inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "assessed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "All current scratch files, including preexisting PlantDoc dependencies and all generated retrain artifacts.",
        "method": "Path-level assessment using producing test/profile lineage and artifact role; retained evidence is hashed. Synthetic pytest contents are not disease data.",
        "total_files": len(records),
        "labels": dict(labels),
        "changed_or_new_files": len(changes),
        "changed_or_new_labels": dict(Counter(row["label"] for row in changes)),
        "changed_or_new_bytes": sum(row["size_bytes"] for row in changes),
        "disposable_bytes": sum(
            row["size_bytes"] for row in records if row["label"] == "DISPOSABLE"
        ),
        "missing_since_snapshot": missing,
        "review_required": [
            row["path"] for row in records if row["label"] == "REVIEW_REQUIRED"
        ],
        "files_deleted_or_renamed_by_assessment": 0,
        "inventory_sha256": hashlib.sha256(
            (REPORT / "scratch-inventory.csv").read_bytes()
        ).hexdigest(),
    }
    (REPORT / "scratch-assessment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
