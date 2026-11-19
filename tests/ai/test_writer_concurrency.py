"""A second writer must never overwrite a split or training run in progress."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ai import split, train
from tests.ai.test_split import dataset as dataset  # noqa: PLC0414
from tests.ai.test_split import grouped_rows as grouped_rows  # noqa: PLC0414
from tests.ai.test_train import config as config  # noqa: PLC0414
from tests.ai.test_train import resume_case as resume_case  # noqa: PLC0414
from tests.ai.test_train import sealed_images as sealed_images  # noqa: PLC0414


def test_split_rejects_second_publisher(
    dataset: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, review = dataset
    replace = os.replace
    competed = False

    def race(src: Path, dst: Path) -> None:
        nonlocal competed
        if dst == root / "manifest.csv" and not competed:
            competed = True
            with pytest.raises(FileExistsError, match="writer"):
                split.prepare_dataset(root, review, seed=13)
        replace(src, dst)

    monkeypatch.setattr(os, "replace", race)
    split.prepare_dataset(root, review, seed=42)
    assert competed
    assert json.loads((root / "split.json").read_text())["seed"] == 42
    assert split.verify_seal(root)["technical_gate"] == "pass"
    assert not list(root.glob("*.lock"))


@pytest.mark.parametrize("resuming", [False, True])
def test_training_rejects_second_writer(
    resume_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resuming: bool,
) -> None:
    out = tmp_path / "shared-run"
    resume = None
    if resuming:
        train.run_training(resume_case, out=out)
        resume = out / "last.pth"
    competed = False

    def compete() -> None:
        nonlocal competed
        if not competed:
            competed = True
            with pytest.raises(FileExistsError, match="writer"):
                train.run_training(resume_case, resume=resume, out=out)

    if resuming:
        publish = train._publish_training

        def race_resume(directory: Path, checkpoint: dict[str, Any]) -> Path:
            compete()
            return publish(directory, checkpoint)

        monkeypatch.setattr(train, "_publish_training", race_resume)
    else:
        save = train._atomic_save

        def race_new(path: Path, payload: dict[str, Any]) -> None:
            compete()
            save(path, payload)

        monkeypatch.setattr(train, "_atomic_save", race_new)

    train.run_training(resume_case, resume=resume, out=out)
    assert competed
    assert (out / "last.pth").is_file()
    assert not list(tmp_path.glob("*.lock"))
