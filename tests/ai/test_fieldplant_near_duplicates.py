import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

audit = import_module(
    "docs.research.yolo26s-field-colab-2026-11-10.fieldplant_near_duplicates"
)


def test_detects_resized_rotated_recompressed_and_mirrored_frames(tmp_path: Path) -> None:
    original = Image.new("RGB", (320, 240), (32, 74, 21))
    draw = ImageDraw.Draw(original)
    draw.ellipse((18, 4, 170, 128), fill=(204, 55, 19))
    draw.polygon([(80, 224), (289, 193), (249, 10)], fill=(19, 163, 53))
    draw.rectangle((207, 132, 319, 181), fill=(141, 84, 205))
    altered = ImageOps.mirror(original).transpose(Image.Transpose.ROTATE_90)
    jpeg = tmp_path / "transformed.jpg"
    altered.resize((155, 207), Image.Resampling.LANCZOS).save(jpeg, quality=43)
    distractor = Image.new("RGB", (240, 320), (128, 128, 128))
    with Image.open(jpeg) as image:
        candidates = [audit.signatures(item, (None,))[0] for item in (image, distractor)]
    matches = audit.compare(
        audit.signatures(original, audit.TRANSFORMS),
        np.array([item[0] for item in candidates], dtype=np.uint64),
        np.stack([item[1] for item in candidates]),
    )
    assert set(matches) == {0}
    assert matches[0][1] < 12
    assert matches[0][2] == "TRANSPOSE"


def test_closure_propagates_protected_and_audit_flags_across_candidates(tmp_path: Path) -> None:
    def source(id_: str, keys: list[tuple[str, str]], flags: tuple[str, ...] = ()) -> Any:
        return audit.Source(tmp_path / id_, id_, set(keys), set(), set(flags))

    sources = {
        "f0": source("f0", [("fieldplant", "g0")]),
        "f1": source("f1", [("fieldplant", "g0")], ("prior_visual_audit",)),
        "f2": source("f2", [("fieldplant", "g2")]),
        "f3": source("f3", []),
        "pd0": source("pd0", []),
        "pv0": source("pv0", [("pv_component", "g4")], ("baseline_exposure",)),
        "pv1": source("pv1", [("pv_component", "g4")], ("protected_split",)),
    }
    groups = audit.candidate_groups(
        sources,
        {"f0": "g0", "f1": "g0", "f2": "g2", "f3": "g3"},
        [("pd0", "pv0")],
        [{"left": "f0", "right": "f2"}, {"left": "f2", "right": "pd0"}],
    )
    assert groups == [{
        "fieldplant_records": ["f0", "f1", "f2"],
        "prior_fieldplant_groups": ["g0", "g2"],
        "exclusion_witnesses": {
            "prior_visual_audit": "f1", "baseline_exposure": "pv0", "protected_split": "pv1"
        },
        "new_exclusion_flags_by_record": {
            "f0": ["baseline_exposure", "prior_visual_audit", "protected_split"],
            "f1": ["baseline_exposure", "protected_split"],
            "f2": ["baseline_exposure", "prior_visual_audit", "protected_split"],
        },
    }]


def test_hash_mismatch_aborts_and_missing_reference_is_counted(tmp_path: Path) -> None:
    path = tmp_path / "original.png"
    Image.new("RGB", (16, 10), (4, 60, 8)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = tmp_path / "missing.png"
    sources = {"field": audit.Source(path, digest), "external": audit.Source(missing, digest)}
    fingerprints, coverage, unavailable = audit.collect_signatures(sources, {"field": "field"})
    assert set(fingerprints) == {"field"}
    assert coverage == {"byte_verified": 1}
    assert unavailable == {"external": "FileNotFoundError"}
    missing.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="Source byte hash changed: external"):
        audit.collect_signatures(sources, {"field": "field"})


def test_large_jpeg_reference_uses_bounded_preview_without_relaxing_training_limit(tmp_path: Path) -> None:
    path = tmp_path / "24mp.jpg"
    with Image.new("RGB", (6000, 4000), (20, 120, 90)) as original:
        original.save(path, quality=80)
    data = path.read_bytes()
    source = audit.Source(path, hashlib.sha256(data).hexdigest())
    default, _, unavailable = audit.collect_signatures({"external": source}, {})
    assert default == {} and unavailable == {"external": "pixel_limit"}
    fingerprints, coverage, unavailable = audit.collect_signatures({"external": source}, {}, jpeg_reference_previews=True)
    assert set(fingerprints) == {"external"} and unavailable == {}
    assert coverage["jpeg_reference_previews"] == 1
    with audit.bounded_jpeg_preview(data) as preview:
        assert preview.width * preview.height <= 2_000_000
    with pytest.raises(ValueError, match="pixel_limit"):
        audit.collect_signatures({"field": source}, {"field": "field"}, jpeg_reference_previews=True)
