"""Audit a pilot's submitted clip groups and keyframe fingerprints without decoding."""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

SPLITS = ("train", "dev", "holdout")
MAX_ROWS = 2_000
MAX_INPUT_BYTES = 4_000_000
NEAR_DISTANCE = 6


def token(value: Any, pattern: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"invalid {field}")
    return value


def audit_split(
    clips: list[dict[str, Any]], frames: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(clips, list) or not isinstance(frames, list):
        raise TypeError("clips and frames must be lists")
    if len(clips) > MAX_ROWS or len(frames) > MAX_ROWS:
        raise ValueError("pilot exceeds the audit row limit")
    ids = r"[a-z0-9][a-z0-9:_-]{0,127}"
    sha = r"[a-f0-9]{64}"
    clip_index: dict[str, tuple[str, int]] = {}
    membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    clips_per_split = dict.fromkeys(SPLITS, 0)
    for row in clips:
        clip = token(row["clip_id"], ids, "clip_id")
        split, count = row["split"], row["keyframe_count"]
        plants = row["plant_ids"]
        if clip in clip_index:
            raise ValueError(f"duplicate clip: {clip}")
        if split not in SPLITS or type(count) is not int or not 1 <= count <= MAX_ROWS:
            raise ValueError(f"invalid split or keyframe count: {clip}")
        if not isinstance(plants, list):
            raise TypeError(f"invalid plant_ids: {clip}")
        if not plants and row.get("reviewed_negative") is not True:
            raise ValueError(f"missing plant_ids without reviewed_negative: {clip}")
        clip_index[clip] = (split, count)
        clips_per_split[split] += 1
        for field in ("site_id", "session_id", "source_sha256"):
            value = token(row[field], sha if field == "source_sha256" else ids, field)
            membership[field, value].add(split)
        for plant in plants:
            membership["plant_ids", token(plant, ids, "plant_ids")].add(split)

    seen: set[tuple[str, int]] = set()
    frame_counts: Counter[str] = Counter()
    fingerprints = []
    for row in frames:
        clip = token(row["clip_id"], ids, "clip_id")
        if clip not in clip_index:
            raise ValueError(f"unknown clip: {clip}")
        index = row["frame_index"]
        if type(index) is not int or index < 0:
            raise ValueError(f"invalid frame_index: {clip}")
        key = (clip, index)
        if key in seen:
            raise ValueError(f"duplicate frame: {key}")
        seen.add(key)
        split, _ = clip_index[clip]
        frame_counts[clip] += 1
        pixel = token(row["pixel_sha256"], sha, "pixel_sha256")
        dhash = int(token(row["dhash64"], r"[a-f0-9]{16}", "dhash64"), 16)
        membership["pixel_sha256", pixel].add(split)
        fingerprints.append((key, split, pixel, dhash))

    conflicts: list[dict[str, Any]] = []
    conflict_counts: Counter[str] = Counter()

    def record(kind: str, **detail: Any) -> None:
        conflict_counts[kind] += 1
        if len(conflicts) < 50:
            conflicts.append({"kind": kind, **detail})

    for (kind, value), splits in sorted(membership.items()):
        if len(splits) > 1:
            record(kind, value=value, splits=sorted(splits))
    for (a, left, a_pixel, a_hash), (b, right, b_pixel, b_hash) in combinations(
        fingerprints, 2
    ):
        if left != right and a_pixel != b_pixel:
            distance = (a_hash ^ b_hash).bit_count()
            if distance <= NEAR_DISTANCE:
                record("near_duplicate", frames=[a, b], distance=distance)

    missing = sorted(
        clip for clip, (_, count) in clip_index.items() if frame_counts[clip] != count
    )
    status = "passed"
    if not clips:
        status = "not_run_no_clips"
    elif conflict_counts:
        status = "failed_overlap"
    elif missing or not all(clips_per_split.values()):
        status = "incomplete_coverage"
    return {
        "status": status,
        "clips_per_split": clips_per_split,
        "keyframes_checked": len(frames),
        "keyframe_count_mismatch": missing,
        "near_distance_inclusive": NEAR_DISTANCE,
        "cross_split_conflict_counts": dict(conflict_counts),
        "cross_split_conflicts": conflicts,
    }


def main() -> int:
    try:
        content = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(content) > MAX_INPUT_BYTES:
            raise ValueError("audit input exceeds byte limit")
        payload = json.loads(content)
        result = audit_split(payload["clips"], payload["frames"])
    except (ValueError, TypeError, KeyError) as error:
        print(f"Invalid split metadata: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
