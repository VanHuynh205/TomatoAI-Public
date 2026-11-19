from importlib import import_module
from typing import Any

import pytest


def run_audit(
    clips: list[dict[str, Any]], frames: list[dict[str, Any]]
) -> dict[str, Any]:
    audit = import_module("docs.progress.yoloe-video.split_audit")
    result: dict[str, Any] = audit.audit_split(clips, frames)
    return result


@pytest.fixture
def sample() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits = ("train", "dev", "holdout")
    clips = [
        {
            "clip_id": split,
            "split": split,
            "site_id": f"{split}-site",
            "session_id": f"{split}-session",
            "plant_ids": [f"{split}-plant"],
            "source_sha256": str(i) * 64,
            "keyframe_count": 1,
        }
        for i, split in enumerate(splits, 1)
    ]
    frames = [
        {
            "clip_id": split,
            "frame_index": 0,
            "pixel_sha256": str(i + 4) * 64,
            "dhash64": hash_value,
        }
        for i, (split, hash_value) in enumerate(
            zip(
                splits,
                ("0000000000000000", "ffffffff00000000", "00000000ffffffff"),
                strict=True,
            )
        )
    ]
    return clips, frames


def test_empty_is_not_an_audit_pass() -> None:
    assert run_audit([], [])["status"] == "not_run_no_clips"


def test_independent_splits_pass(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    result = run_audit(*sample)
    assert result["status"] == "passed"
    assert result["clips_per_split"] == {"train": 1, "dev": 1, "holdout": 1}
    assert result["cross_split_conflicts"] == []


@pytest.mark.parametrize(
    "field", ["site_id", "session_id", "plant_ids", "source_sha256"]
)
def test_shared_group_or_clip_bytes_are_rejected(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]], field: str
) -> None:
    clips, frames = sample
    clips[-1][field] = clips[0][field]
    result = run_audit(clips, frames)
    assert result["status"] == "failed_overlap"
    assert {item["kind"] for item in result["cross_split_conflicts"]} == {field}


@pytest.mark.parametrize(
    "field,value,kind",
    [
        ("pixel_sha256", "4" * 64, "pixel_sha256"),
        ("dhash64", "000000000000003f", "near_duplicate"),
    ],
)
def test_cross_split_fingerprints_are_rejected(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    field: str,
    value: str,
    kind: str,
) -> None:
    clips, frames = sample
    frames[-1][field] = value
    result = run_audit(clips, frames)
    assert result["status"] == "failed_overlap"
    assert {item["kind"] for item in result["cross_split_conflicts"]} == {kind}


def test_missing_holdout_or_fingerprint_does_not_pass(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    clips, frames = sample
    assert run_audit(clips[:-1], frames[:-1])["status"] == "incomplete_coverage"
    assert run_audit(clips, frames[:-1])["status"] == "incomplete_coverage"


def test_unknown_clip_and_duplicate_frame_are_invalid(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    clips, frames = sample
    with pytest.raises(ValueError, match="unknown clip"):
        run_audit(clips, frames + [frames[0] | {"clip_id": "unknown"}])
    with pytest.raises(ValueError, match="duplicate frame"):
        run_audit(clips, frames + [frames[0]])


def test_only_reviewed_negative_may_omit_plant_group(
    sample: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    clips, frames = sample
    clips[-1]["plant_ids"] = []
    with pytest.raises(ValueError, match="plant_ids"):
        run_audit(clips, frames)
    clips[-1]["reviewed_negative"] = True
    assert run_audit(clips, frames)["status"] == "passed"
