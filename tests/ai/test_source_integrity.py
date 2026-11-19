"""M1 must record hashes for the same immutable source image."""

import io
import os
from typing import BinaryIO

import pytest
from PIL import Image

from ai import data
from tests.ai.conftest import AuditCase


def test_audit_excludes_source_changed_during_decode(
    audit_case: AuditCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = audit_case
    case.rows = case.rows[:1]
    case.write_rows()
    original = case.raw.read_bytes()
    before = case.raw.stat()
    decode = data._decode_rgb

    def mutate(stream: BinaryIO) -> Image.Image:
        with Image.new("RGB", (8, 8), "red") as image, io.BytesIO() as buffer:
            image.save(buffer, "PNG")
            case.raw.write_bytes(buffer.getvalue())
        try:
            return decode(stream)
        finally:
            case.raw.write_bytes(original)
            os.utime(case.raw, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    monkeypatch.setattr(data, "_decode_rgb", mutate)
    counts = data.audit_manifest(case.csv, case.root, case.out, case.provenance)

    assert counts == {"accepted": 0, "excluded": 1}
    assert "source_changed" in (case.out / "exclusions.csv").read_text()
    assert case.raw.read_bytes() == original
