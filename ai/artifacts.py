"""Bounded offline model bundles; provenance hashes are not signatures."""

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai.contracts import CLASS_ID_PATTERN
from ai.models import build_model
from ai.preprocessing import build_transform

_METADATA_LIMIT = 1_000_000
_WEIGHTS_LIMIT = 256_000_000
_FIELDS = {
    "schema_version",
    "architecture",
    "model_version",
    "class_ids",
    "preprocessing",
    "dataset_version",
    "dataset_sha256",
    "split_hash",
    "weights_sha256",
    "threshold",
    "class_map_sha256",
    "validation_decision_sha256",
    "artifact_kind",
}


@dataclass
class ModelBundle:
    model: nn.Module
    metadata: dict[str, Any]


def _local(path: Path) -> Path:
    value = str(path)
    if (
        not value.strip()
        or "://" in value
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]+:", value) is not None
        or "\x00" in value
        or value.replace("\\", "/").startswith("//")
    ):
        raise ValueError("bundle path must be local")
    result = Path(os.path.abspath(path))
    if result.resolve() != result:
        raise ValueError("bundle path must not traverse symlinks or junctions")
    return result


def _read(path: Path, limit: int) -> bytes:
    path = _local(path)
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError(f"{path.name} size limit")
        content = stream.read(limit + 1)
    if not 0 < len(content) <= limit:
        raise ValueError(f"{path.name} size limit")
    return content


def _json(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid metadata JSON") from error
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")  # noqa: TRY004
    return value


def _name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) is not None
    )


def _threshold(value: Any) -> bool:
    return value is None or (
        type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)
    )


def _metadata(meta: dict[str, Any]) -> None:
    if set(meta) != _FIELDS:
        raise ValueError("metadata fields mismatch")
    if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
        raise ValueError("unsupported bundle schema")
    if meta["architecture"] not in ("efficientnet_b0", "resnet50"):
        raise ValueError("unsupported architecture")
    for field in ("model_version", "dataset_version"):
        if not _name(meta[field]):
            raise ValueError(f"invalid {field}")
    ids = meta["class_ids"]
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 1000
        or any(
            not isinstance(x, str) or re.fullmatch(CLASS_ID_PATTERN, x) is None
            for x in ids
        )
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid class_ids")
    if meta["artifact_kind"] not in ("trained", "fixture"):
        raise ValueError("invalid artifact_kind")
    if not _threshold(meta["threshold"]):
        raise ValueError("invalid threshold")
    for field in (
        "weights_sha256",
        "class_map_sha256",
        "dataset_sha256",
        "split_hash",
        "validation_decision_sha256",
    ):
        if (
            not isinstance(meta[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", meta[field]) is None
        ):
            raise ValueError(f"invalid {field}")
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != meta["class_map_sha256"]:
        raise ValueError("class_map_sha256 mismatch")
    profile = meta["preprocessing"]
    if (
        not isinstance(profile, dict)
        or set(profile) != {"profile_version", "spec"}
        or profile["profile_version"] != "whole-frame-v1"
    ):
        raise ValueError("unsupported preprocessing profile")
    try:
        build_transform(profile["spec"], training=False)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError("invalid preprocessing spec") from error


def load_bundle(bundle_dir: Path, *, require_trained: bool = False) -> ModelBundle:
    """Load two local files; release callers must require trained artifacts."""
    if type(require_trained) is not bool:
        raise ValueError("require_trained must be bool")
    directory = _local(bundle_dir)
    meta = _json(_read(directory / "metadata.json", _METADATA_LIMIT))
    _metadata(meta)
    if require_trained and meta["artifact_kind"] != "trained":
        raise ValueError("fixture bundle forbidden in release")
    content = _read(directory / "model.pth", _WEIGHTS_LIMIT)
    if hashlib.sha256(content).hexdigest() != meta["weights_sha256"]:
        raise ValueError("weights_sha256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(info.file_size for info in archive.infolist()) > _WEIGHTS_LIMIT:
                raise ValueError("weights expanded size limit")
        state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (
        zipfile.BadZipFile,
        pickle.UnpicklingError,
        RuntimeError,
        EOFError,
    ) as error:
        raise ValueError("invalid safe weights") from error
    model = build_model(meta["architecture"], len(meta["class_ids"]))
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("state_dict keys mismatch")
    for key, tensor in expected.items():
        saved = state[key]
        if (
            not isinstance(saved, torch.Tensor)
            or saved.layout != torch.strided
            or saved.shape != tensor.shape
            or saved.dtype != tensor.dtype
        ):
            raise ValueError(f"state_dict tensor mismatch: {key}")
        if not bool(torch.isfinite(saved).all()):
            raise ValueError(f"state_dict non-finite tensor: {key}")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return ModelBundle(model=model, metadata=meta)


def export_bundle(
    run_dir: Path, output_dir: Path, model_version: str, threshold: float | None
) -> Path:
    """Export a selected M5 best checkpoint without overwriting an existing bundle."""
    # Training dependencies stay out of the runtime load/predict path.
    from ai.experiments import _completed_run, summarize_validation

    directory, output = _local(run_dir), _local(output_dir)
    if output.exists():
        raise FileExistsError(f"bundle already exists: {output}")
    if not _name(model_version):
        raise ValueError("invalid model_version")
    decision_bytes = _read(directory / "decision.json", _METADATA_LIMIT)
    decision = _json(decision_bytes)
    try:
        ids = [row["run_id"] for row in decision["compared_runs"]]
        if (
            type(decision["schema_version"]) is not int
            or decision["schema_version"] != 1
            or decision["architecture"]
            not in ("efficientnet_b0", "resnet50")
            or decision["run_id"] != directory.name
            or len(ids) != 2
            or any(not _name(value) for value in ids)
            or len(set(ids)) != 2
            or directory.name not in ids
        ):
            raise ValueError("invalid selection decision")
        if not _threshold(threshold) or threshold != decision["threshold"]:
            raise ValueError("threshold must match saved decision")
    except (KeyError, TypeError) as error:
        raise ValueError("invalid selection decision") from error
    summary = summarize_validation([_local(directory.parent / value) for value in ids])
    expected = summary["decisions"].get(decision["architecture"])
    if json.dumps(decision, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("saved decision differs from verified validation selection")
    protected = [directory.parent, *(Path(value) for value in summary["data"].values())]
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("output must be outside runs, dataset and raw source")
    _, best, _, metadata_hash = _completed_run(directory)
    if (
        metadata_hash != decision["metadata_sha256"]
        or best["epoch"] != decision["best_epoch"]
        or hashlib.sha256(_read(directory / "best.pth", 512_000_000)).hexdigest()
        != decision["best_sha256"]
        or _read(directory / "decision.json", _METADATA_LIMIT) != decision_bytes
    ):
        raise ValueError("run changed after validation decision verification")
    config = best["config"]
    meta = {
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
        torch.save(best["model"], stage / "model.pth")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--model-version", required=True)
    export.add_argument("--decision", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if _local(args.decision) != _local(args.run) / "decision.json":
                raise ValueError(
                    "decision must be the selected run's saved decision.json"
                )
            decision = _json(_read(args.decision, _METADATA_LIMIT))
            if "threshold" not in decision:
                raise ValueError("decision missing threshold")
            output = export_bundle(
                args.run, args.out, args.model_version, decision["threshold"]
            )
            event = "bundle_exported"
        else:
            load_bundle(args.bundle, require_trained=args.release)
            output, event = args.bundle, "bundle_verified"
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"event": event, "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
