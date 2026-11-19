"""Serialize cooperative writers without changing checkpoint or dataset formats."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_writer(path: Path) -> Iterator[None]:
    # Leave the marker on abrupt process death so recovery can inspect the output.
    try:
        with path.open("xb"):
            pass
    except FileExistsError as error:
        raise FileExistsError(
            f"writer already active or interrupted: {path}"
        ) from error
    try:
        yield
    finally:
        path.unlink()
