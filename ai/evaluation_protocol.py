"""Freeze M6 releases and reserve a single TEST read per model on local storage."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.artifacts import (
    _METADATA_LIMIT,
    _WEIGHTS_LIMIT,
    _json,
    _local,
    _read,
    load_bundle,
)
from ai.data import _sha256
from ai.split import ARTIFACTS, _hash, _path

_SOURCE_DIR = Path(__file__).resolve().parent
_LEDGER = "test-open-ledger"


def _encoded(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


def _digest(path: Path, limit: int = 512_000_000) -> str:
    path = _local(path)
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"missing or oversized protocol input: {path}")
    return _sha256(path)


def _relative(path: Path, protocol: Path) -> str:
    return Path(os.path.relpath(path, protocol.parent)).as_posix()


def _split_anchor(dataset: Path, split_hash: str) -> Path:
    if not _hash(split_hash):
        raise ValueError("invalid locked split hash")
    return _local(dataset.parent / ".test-protocols" / f"{split_hash}.json")


def _source_hashes() -> dict[str, str]:
    sources = {p.name: _sha256(p) for p in sorted(_SOURCE_DIR.glob("*.py"))}
    if "evaluate.py" not in sources:
        raise ValueError("missing evaluation source evaluate.py")
    return sources


def _write_once(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _check_data(dataset: Path, data: dict[str, Any]) -> None:
    if _digest(dataset / "seal.json", _METADATA_LIMIT) != data["seal_sha256"]:
        raise ValueError("dataset seal changed")
    # Hash sealed artifacts as opaque bytes; never enumerate TEST samples here.
    for name, expected in data["hashes"].items():
        if _digest(_path(dataset, name)) != expected:
            raise ValueError(f"sealed data changed: {name}")


def _bundle_record(
    bundle: Path, decision: Path, dataset: Path, data: dict[str, Any], protocol: Path
) -> dict[str, Any]:
    meta = load_bundle(bundle, require_trained=True).metadata
    raw_meta = _read(bundle / "metadata.json", _METADATA_LIMIT)
    if _json(raw_meta) != meta:
        raise ValueError("bundle metadata changed during lock")
    if (
        meta["dataset_version"] != dataset.name
        or meta["dataset_sha256"] != data["seal_sha256"]
        or meta["split_hash"] != data["hashes"]["split.json"]
        or meta["class_ids"] != data["class_ids"]
    ):
        raise ValueError("bundle and sealed dataset identities disagree")
    raw_decision = _read(decision, _METADATA_LIMIT)
    if hashlib.sha256(raw_decision).hexdigest() != meta["validation_decision_sha256"]:
        raise ValueError("bundle and validation decision digests disagree")
    selected = _json(raw_decision)
    identity = selected.get("identity")
    if (
        decision.name != "decision.json"
        or type(selected.get("schema_version")) is not int
        or selected["schema_version"] != 1
        or selected.get("architecture") != meta["architecture"]
        or selected.get("run_id") != decision.parent.name
        or selected.get("threshold") != meta["threshold"]
        or type(selected.get("best_epoch")) is not int
        or selected["best_epoch"] < 1
        or not isinstance(selected.get("selection_rule"), str)
        or not isinstance(selected.get("threshold_policy"), dict)
        or not isinstance(identity, dict)
        or identity.get("seal_sha256") != data["seal_sha256"]
        or identity.get("split_hash") != meta["split_hash"]
        or identity.get("class_ids") != data["class_ids"]
        or _digest(decision.with_name("best.pth")) != selected.get("best_sha256")
    ):
        raise ValueError("invalid validation checkpoint or selection decision")
    return {
        "bundle_path": _relative(bundle, protocol),
        "decision_path": _relative(decision, protocol),
        "metadata_sha256": hashlib.sha256(raw_meta).hexdigest(),
        **{
            key: meta[key]
            for key in (
                "model_version",
                "architecture",
                "weights_sha256",
                "class_map_sha256",
                "preprocessing",
                "validation_decision_sha256",
                "threshold",
            )
        },
        **{
            key: selected[key]
            for key in (
                "run_id",
                "best_epoch",
                "best_sha256",
                "selection_rule",
                "threshold_policy",
            )
        },
    }


def create_protocol(
    bundle_dirs: list[Path], dataset_dir: Path, protocol: Path, *, decisions: list[Path]
) -> Path:
    """Pair each bundle with its M5 decision in order. Publish only once."""
    protocol, dataset = _local(protocol), _local(dataset_dir)
    ledger = _local(protocol.parent / _LEDGER)
    if protocol == ledger:
        raise ValueError("protocol path collides with TEST ledger")
    if protocol.exists() or ledger.exists():
        raise FileExistsError("protocol or TEST ledger already exists")
    if len(bundle_dirs) != 2 or len(decisions) != 2:
        raise ValueError("two release bundles and corresponding decisions required")
    bundles, paths = list(map(_local, bundle_dirs)), list(map(_local, decisions))
    seal_bytes = _read(dataset / "seal.json", _METADATA_LIMIT)
    seal = _json(seal_bytes)
    hashes = seal.get("hashes")
    if (
        type(seal.get("schema_version")) is not int
        or seal["schema_version"] != 1
        or seal.get("technical_gate") != "pass"
        or not isinstance(hashes, dict)
        or not set(ARTIFACTS).issubset(hashes)
    ):
        raise ValueError("sealed dataset with passing technical gate required")
    data = {
        "path": _relative(dataset, protocol),
        "seal_sha256": hashlib.sha256(seal_bytes).hexdigest(),
        "hashes": hashes,
    }
    _check_data(dataset, data)
    mapping = _json(_read(dataset / "label_map.json", _METADATA_LIMIT))
    data["class_ids"] = [item["class_id"] for item in mapping["classes"]]
    provenance = _json(_read(dataset / "provenance.json", _METADATA_LIMIT))
    raw_root = (dataset / provenance["source_root"]).resolve()
    protected = [dataset, raw_root, *bundles, *(p.parent for p in paths)]
    anchor = _split_anchor(dataset, hashes["split.json"])
    if protocol.is_relative_to(anchor.parent):
        raise ValueError("protocol path collides with the split lock registry")
    if any(
        path.is_relative_to(root) for root in protected for path in (protocol, anchor)
    ):
        raise ValueError(
            "protocol and split lock must be outside data, raw images, runs and bundles"
        )
    records = [
        _bundle_record(bundle, decision, dataset, data, protocol)
        for bundle, decision in zip(bundles, paths, strict=True)
    ]
    if {r["architecture"] for r in records} != {"efficientnet_b0", "resnet50"} or len(
        {r["model_version"] for r in records}
    ) != 2:
        raise ValueError("distinct EfficientNet-B0 and ResNet50 releases required")
    content = _encoded(
        {
            "schema_version": 1,
            "split": "test",
            "locked_at": datetime.now(UTC).isoformat(),
            "dataset": data,
            "bundles": sorted(records, key=lambda r: r["architecture"]),
            "sources_sha256": _source_hashes(),
            "checkpoint_rule": "max validation macro-F1, then min validation loss, then earliest epoch",
            "abstention_rule": "accept iff threshold is set and top-1 softmax >= threshold",
            "open_rule": "one reservation per architecture; a failed evaluation consumes its reservation",
        }
    )
    binding = _encoded(
        {
            "protocol": str(protocol),
            "protocol_sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    anchor.parent.mkdir(exist_ok=True)
    _write_once(anchor, binding)
    protocol.parent.mkdir(parents=True, exist_ok=True)
    ledger.mkdir()
    pending = ledger / "lock.pending"
    _write_once(pending, binding)
    _write_once(protocol, content)
    # The split stays claimed after a failed write. Only this rename opens the gate.
    pending.rename(ledger / "lock.json")
    return protocol


def verify_protocol(
    protocol: Path, bundle_dir: Path, dataset_dir: Path
) -> dict[str, Any]:
    """Recheck both releases and all locked bytes without spending a reservation."""
    protocol, bundle, dataset = (
        _local(protocol),
        _local(bundle_dir),
        _local(dataset_dir),
    )
    try:
        content = _read(protocol, _METADATA_LIMIT)
        binding = {
            "protocol": str(protocol),
            "protocol_sha256": hashlib.sha256(content).hexdigest(),
        }
        if (
            _json(_read(protocol.parent / _LEDGER / "lock.json", _METADATA_LIMIT))
            != binding
        ):
            raise ValueError("protocol differs from immutable ledger lock")
        record = _json(content)
        anchor = _split_anchor(dataset, record["dataset"]["hashes"]["split.json"])
        if _json(_read(anchor, _METADATA_LIMIT)) != binding:
            raise ValueError("protocol differs from the holdout split lock")
        if record["dataset"]["path"] != _relative(dataset, protocol) or all(
            r["bundle_path"] != _relative(bundle, protocol) for r in record["bundles"]
        ):
            raise ValueError("dataset or bundle outside locked protocol")
        if _source_hashes() != record["sources_sha256"]:
            raise ValueError("evaluation source changed after lock")
        _check_data(dataset, record["dataset"])
        for r in record["bundles"]:
            release = protocol.parent / r["bundle_path"]
            decision = protocol.parent / r["decision_path"]
            for path, key, limit in (
                (release / "metadata.json", "metadata_sha256", _METADATA_LIMIT),
                (release / "model.pth", "weights_sha256", _WEIGHTS_LIMIT),
                (decision, "validation_decision_sha256", _METADATA_LIMIT),
                (decision.with_name("best.pth"), "best_sha256", 512_000_000),
            ):
                if _digest(path, limit) != r[key]:
                    raise ValueError(f"locked {key} changed: {path.name}")
        return record
    except (FileNotFoundError, KeyError, TypeError) as error:
        raise ValueError("missing or invalid locked evaluation protocol") from error


def reserve_test_open(protocol: Path, bundle_dir: Path, dataset_dir: Path) -> Path:
    """Verify before exclusively reserving. Never roll back after an IO or evaluation failure."""
    protocol = _local(protocol)
    record = verify_protocol(protocol, bundle_dir, dataset_dir)
    relative = _relative(_local(bundle_dir), protocol)
    model = next(r for r in record["bundles"] if r["bundle_path"] == relative)
    receipt = _local(protocol.parent / _LEDGER / f"{model['architecture']}.json")
    try:
        _write_once(
            receipt,
            _encoded(
                {
                    "status": "reserved",
                    "reserved_at": datetime.now(UTC).isoformat(),
                    "model_version": model["model_version"],
                    "weights_sha256": model["weights_sha256"],
                    "protocol_sha256": hashlib.sha256(_encoded(record)).hexdigest(),
                }
            ),
        )
    except FileExistsError as error:
        raise ValueError(f"TEST already opened for {model['model_version']}") from error
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("--bundle", type=Path, action="append", required=True)
    lock.add_argument("--decision", type=Path, action="append", required=True)
    lock.add_argument("--dataset", type=Path, required=True)
    lock.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        verify_protocol(args.protocol, args.bundle, args.dataset)
        print(f"Verified {args.protocol}")
    else:
        print(
            create_protocol(
                args.bundle, args.dataset, args.out, decisions=args.decision
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
