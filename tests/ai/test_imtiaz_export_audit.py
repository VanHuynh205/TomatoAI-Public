"""Malformed source annotations must not silently enter the detection audit."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "docs/research/yolo26s-field-colab-2026-11-10/audit_imtiaz_export.py"
SPEC = importlib.util.spec_from_file_location("imtiaz_audit", PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("row", [b"7 .5 .5 .1 .1", b"1 .5 .5 nan .1", b"1 .5 .5 0 .1", b"1 .5 .5 1.1 .1", b"1 .5 .5 .1"])
def test_rejects_invalid_class_geometry_and_structure(row):
    with pytest.raises(ValueError):
        audit.parse_boxes(row)


def test_preserves_source_labels_boxes_and_unresolved_empty_annotations():
    assert audit.parse_boxes(b"6 0.5 0.5 0.8 0.6\n") == [
        {"source_id": 6, "source_label": "black spot", "xywh_normalized": [0.5, 0.5, 0.8, 0.6]}
    ]
    assert audit.parse_boxes(b"\n") == []
