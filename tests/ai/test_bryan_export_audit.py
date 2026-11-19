"""Metadata checks must not invent disease labels or accept malformed boxes."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "docs/research/yolo26s-field-colab-2026-11-10/audit_bryan_exports.py"
SPEC = importlib.util.spec_from_file_location("bryan_audit", PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_filename_key_only_strips_export_name_artifacts():
    assert audit.filename_key("train/Early-Blight21_jpg.rf.1234.jpg") == audit.filename_key("Early_Blight21.jpg")
    assert audit.filename_key("leaf1.jpg") != audit.filename_key("leaf2.jpg")


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 10, float("nan")], [0, 0, 101, 100], [-1, 0, 3, 3], [1, 2, 3]])
def test_bad_geometry_is_rejected(box):
    assert not audit.valid_box(box, 100, 100)


def test_source_rounding_tolerance_is_bounded():
    assert audit.valid_box([-0.5, 0, 100.5, 100], 100, 100)
    assert not audit.valid_box([-0.5001, 0, 100.5, 100], 100, 100)


def test_detail_selection_does_not_open_new_images_or_accept_changed_identity():
    source = {"fork_member": "seen", "fork_sha256": "pinned"}
    selection = {"records": [source, {"fork_member": "unseen", "fork_sha256": "other"}]}
    review = {"reviewed_images": [{**source, "visual_status": "geometry_candidate", "sample": "1"},
                                  {"fork_member": "rejected", "visual_status": "completeness_unresolved"}]}
    assert audit.geometry_candidates(selection, review) == [{**source, "sample": "1"}]
    review["reviewed_images"][0]["fork_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        audit.geometry_candidates(selection, review)


def test_exposure_closure_follows_transitive_aliases_across_exports():
    rows = [
        {"export": "v63", "member": "c", "filename_key": "alias", "sha256": "third"},
        {"export": "fork", "member": "b", "filename_key": "alias", "sha256": "seen"},
        {"export": "v63", "member": "a", "filename_key": "original", "sha256": "seen"},
        {"export": "fork", "member": "unrelated", "filename_key": "other", "sha256": "other"},
    ]
    seeds = [{"filename_key": "original", "sha256": "seed"}]
    expected = {("v63", "a"), ("fork", "b"), ("v63", "c")}
    assert audit.exposure_closure(rows, seeds) == expected
    assert audit.exposure_closure(list(reversed(rows)), seeds) == expected


def test_family_quarantine_does_not_generalize_to_other_camera_families():
    rows = [
        {"export": "fork", "member": "a", "filename_key": "iphone690c0img1", "sha256": "a"},
        {"export": "fork", "member": "b", "filename_key": "iphoneotherimg1", "sha256": "b"},
    ]
    assert audit.exposure_closure(rows, [], ("iphone690c0",)) == {("fork", "a")}
    with pytest.raises(ValueError, match="Empty"):
        audit.exposure_closure(rows, [], ("",))


def record(boxes, size=512):
    return {"width": size, "height": size, "boxes": [{"bbox_xywh": box, "source_class": label} for box, label in boxes]}


def test_correspondence_requires_unique_geometry_and_consistent_known_labels():
    fork = record([([10, 20, 30, 40], "Other Disease"), ([60, 70, 80, 90], "Leaf Mold")])
    source = record([([60, 70, 80, 90], "Leaf Mold"), ([10, 20, 30, 40], "Septoria")])
    assert audit.pair_boxes(fork, source) == [1, 0]
    source["boxes"][0]["source_class"] = "Early Blight"
    assert audit.pair_boxes(fork, source) is None
    ambiguous = record([([10, 20, 30, 40], "Other Disease"), ([10, 20, 30, 40], "Other Disease")])
    assert audit.pair_boxes(ambiguous, ambiguous) is None


def test_correspondence_rejects_missing_extra_or_displaced_boxes():
    original = record([([10, 20, 30, 40], "Other Disease")])
    assert audit.pair_boxes(original, record([])) is None
    assert audit.pair_boxes(original, record([([13, 20, 30, 40], "Septoria")])) is None
    assert audit.pair_boxes(original, record([([20, 40, 60, 80], "Septoria")], size=1024)) == [0]


def test_pixels_accept_same_image_and_jpeg_but_reject_rotation_and_other_scene():
    import io

    import numpy as np
    from PIL import Image

    y, x = np.indices((192, 256))
    pixels = np.stack(((x + 2 * y) % 256, (2 * x + y) % 256, (x // 32 + y // 24) % 2 * 170), axis=2).astype(np.uint8)
    original = Image.fromarray(pixels)
    reference = original.resize((128, 128), Image.Resampling.BILINEAR)
    assert audit.compare_pixels(original, reference)["exact_resized_pixels"]
    buffer = io.BytesIO()
    reference.save(buffer, format="JPEG", quality=95, subsampling=0)
    with Image.open(io.BytesIO(buffer.getvalue())) as compressed:
        assert audit.compare_pixels(original, compressed)["accepted"]
    assert not audit.compare_pixels(original, reference.transpose(Image.Transpose.ROTATE_180))["accepted"]
    assert not audit.compare_pixels(original, Image.fromarray(np.roll(np.asarray(reference), 24, axis=0)))["accepted"]
    assert not audit.compare_pixels(Image.new("RGB", (100, 100), "white"), Image.new("RGB", (100, 100), "white"))["accepted"]
