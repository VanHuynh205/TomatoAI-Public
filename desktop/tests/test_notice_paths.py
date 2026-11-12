"""Dependency license trees can name upstream tests without shipping test assets."""

from pathlib import Path

import pytest

from desktop.release import _files


def test_upstream_test_license_is_inventoried(tmp_path: Path) -> None:
    license_file = (
        tmp_path
        / "_internal/torch-2.14.1+cpu.dist-info/licenses/third_party/fbgemm/test/LICENSE"
    )
    license_file.parent.mkdir(parents=True)
    license_file.write_text("Original dependency license")
    assert (
        _files(tmp_path)[license_file.relative_to(tmp_path).as_posix()] == license_file
    )


def test_runtime_fixture_asset_is_still_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "_internal/frontend/tests/fixture.png"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="test/fixture"):
        _files(tmp_path)


def test_cpu_torch_profiler_python_is_not_a_cuda_runtime(tmp_path: Path) -> None:
    module = tmp_path / "_internal/torch/profiler/_cupti/cupti_python.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Python support shipped by the CPU wheel")
    assert module.relative_to(tmp_path).as_posix() in _files(tmp_path)
