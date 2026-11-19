from importlib import import_module

import pytest

AUDIT = "docs.progress.yolo-image-recovery.audit_roboflow"


def test_names_match_common_roboflow_rewrites() -> None:
    audit = import_module(AUDIT)
    assert audit.filename_key("1058_0_jpeg-itok-OQkdtxgv_jpg") == audit.filename_key(
        "1058_0.jpeg?itok=OQkdtxgv.jpg"
    )
    assert audit.filename_key("1-36_thumb_png") == audit.filename_key("1-36_thumb.png")
    assert audit.filename_key("Image_1.jpg") != audit.filename_key("Image_2.jpg")


def test_boxes_reject_nonfinite_out_of_bounds_and_bad_class() -> None:
    parse = import_module(AUDIT).parse_classes
    assert parse("0 0.5 0.5 1 1\n3 0.4 0.6 0.2 0.2\n") == {0, 3}
    assert parse("") == set()
    for bad in (
        "0 nan 0.5 0.5 0.5",
        "0 0.9 0.5 0.5 0.5",
        "0 0.5 0.5 0 0.5",
        "10 0.5 0.5 0.5 0.5",
        "0 0.5 0.5 0.5",
    ):
        with pytest.raises(ValueError):
            parse(bad)


def test_near_overlap_checks_full_mirror_and_crop_independently() -> None:
    near = import_module("docs.progress.yolo-image-recovery.near_roboflow")
    reference = (0, (1 << 63) - 1)
    assert near.near_distances({(1 << 63) - 1}, {(1 << 20) - 1}, reference) == (0, 20)
    assert near.near_distances({(1 << 12) - 1}, {15}, reference) == (12, 4)
    assert near.near_distances({(1 << 12) - 1}, {31}, reference) is None
