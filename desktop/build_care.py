"""Export only explicitly allowlisted, source-checked care cards."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.care import _Bundle, _json, _local, _read
from backend.schemas import CareCard, ClassId, NonEmptyText


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: ClassId
    file: str = Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")
    sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")


class _Allowlist(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(strict=True, ge=1, le=1)
    knowledge_version: NonEmptyText = Field(max_length=64)
    allowed_use: Literal["offline_reference_vn"]
    cards: list[_Entry] = Field(min_length=1, max_length=1000)


def export_care(cards_dir: Path, allowlist_path: Path, output_path: Path) -> None:
    directory, allow_path, output = (
        _local(cards_dir),
        _local(allowlist_path),
        _local(output_path),
    )
    if output.is_relative_to(directory) or output == allow_path:
        raise ValueError("output must not replace input cards or allowlist")
    allowed = _Allowlist.model_validate(_json(_read(allow_path)))
    ids = [row.class_id for row in allowed.cards]
    if len(ids) != len(set(ids)) or len({row.file for row in allowed.cards}) != len(
        ids
    ):
        raise ValueError("duplicate allowlist card")
    cards = []
    for entry in allowed.cards:
        content = _read(directory / entry.file)
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError("allowlist card hash mismatch")
        card = CareCard.model_validate(_json(content))
        if card.class_id != entry.class_id:
            raise ValueError("allowlist class_id mismatch")
        cards.append(card)
    bundle = _Bundle(
        schema_version=1,
        knowledge_version=allowed.knowledge_version,
        artifact_kind="curated",
        cards=sorted(cards, key=lambda card: card.class_id),
    )
    content = (
        json.dumps(
            bundle.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
        )
        + "\n"
    ).encode("utf-8")
    if len(content) > 5_000_000:
        raise ValueError("care size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=".care-", suffix=".json", dir=output.parent
    )
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        export_care(args.cards, args.allowlist, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"event": "care_exported", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
