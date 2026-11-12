"""Fresh-clone checks use synthetic inputs, never training data or real weights."""

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from desktop.release import validate_care_export

ROOT = Path(__file__).resolve().parents[1]


def test_care_build_inputs_reproduce_published_catalog() -> None:
    validate_care_export(ROOT / "infra/knowledge", ROOT / "infra/knowledge")


def test_project_license_metadata_and_notices_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text(encoding="utf-8"))
    assert project["project"]["license"] == "Apache-2.0"
    assert package["license"] == lock["packages"][""]["license"] == "Apache-2.0"
    assert "Version 2.0, January 2004" in (ROOT / "LICENSE").read_text()
    assert "Copyright (c) 2026 VanHuynh205" in (ROOT / "NOTICE").read_text()
    assert "MIT License" in (ROOT / "desktop/notices/project-MIT-legacy.txt").read_text()
    assert "CC BY-SA 4.0" in (ROOT / "infra/knowledge/CONTENT-LICENSE.md").read_text(encoding="utf-8")
    assert lock["packages"]["node_modules/react"]["license"] == "MIT"


@pytest.mark.parametrize("name", [
    "MODEL_CARD.md", "DATASET_ATTRIBUTIONS.md", "docs/care/review.md",
    "docs/care/licensed-source-map.md", "docs/contracts/openapi-v1.json",
    "frontend/scripts/check-api-types.mjs", "frontend/src/test/setup.ts",
    "frontend/e2e.config.ts", "frontend/tests/desktop.e2e.ts",
    "frontend/tests/fixtures/expected.json", "desktop/tests/browser_server.py",
    "desktop/tests/fixtures/care.json", "desktop/tests/fixtures/healthy.png",
    "desktop/tests/fixtures/healthy.jpg", "desktop/tests/fixtures/uncertain.png",
    "desktop/tests/fixtures/fake.png", "frontend/dist/index.html",
])
def test_public_developer_and_runtime_inputs_exist(name: str) -> None:
    assert (ROOT / name).is_file(), name


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only bootstrap")
def test_bootstrap_missing_bundle_stops_before_installing(tmp_path: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    shutil.copyfile(ROOT / "Start.ps1", tmp_path / "Start.ps1")
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-File", str(tmp_path / "Start.ps1"), "-CheckOnly"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Missing authorized DINOv2 v1 bundle files" in output
    assert "classifier.safetensors" in output
    assert "model.safetensors" in output
    assert not (tmp_path / ".venv-runtime").exists()
