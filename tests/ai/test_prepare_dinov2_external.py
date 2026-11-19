import hashlib
import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from ai import prepare_dinov2_external as external


def test_cached_manifest_must_match_mendeley_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"cached test image"
    digest = hashlib.sha256(content).hexdigest()
    cache = tmp_path / "raw/tom2024"
    cache.mkdir(parents=True)
    (cache / f"{digest}.jpg").write_bytes(content)
    source_id = "bebe0b82-3445-4ac9-8ad1-21f402d0c085"
    url = f"https://data.mendeley.com/public-files/datasets/3d4yg89rtr/files/{source_id}/file_downloaded"
    row = {"source_id": source_id, "source_file": "leaf.jpg", "original_sha256": digest, "download_url": url}
    manifest = tmp_path / external.TOM_MANIFEST
    external.write_csv(manifest, [row])
    (tmp_path / "taiwan.7z").write_bytes(b"archive")
    monkeypatch.setattr(external, "TAIWAN_SHA", hashlib.sha256(b"archive").hexdigest())
    monkeypatch.setattr("ai.prepare_dinov2_external.subprocess.run", Mock())
    offline = Mock(side_effect=AssertionError("cached test requires no network"))
    monkeypatch.setattr(external, "fetch", offline)
    external.fetch_sources(tmp_path)
    for field, value in (("download_url", external.TAIWAN_URL), ("original_sha256", "incomplete")):
        manifest.unlink()
        external.write_csv(manifest, [{**row, field: value}])
        with pytest.raises(ValueError, match="TOM2024"):
            external.fetch_sources(tmp_path)
    offline.assert_not_called()


def test_fetch_limits_body_before_accepting_response(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock(side_effect=[io.BytesIO(b"[{}]"), io.BytesIO(b"[{}] ")])
    monkeypatch.setattr(external, "urlopen", reader)
    monkeypatch.setattr(external, "TOM2024_MAX_BYTES", 4)
    url = "https://data.mendeley.com/public-api/datasets/"
    assert external.fetch(url, listing=True) == b"[{}]"
    with pytest.raises(ValueError, match="size limit"):
        external.fetch(url, listing=True)
