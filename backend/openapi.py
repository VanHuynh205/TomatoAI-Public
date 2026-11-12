"""Export the same route schemas as the runtime without loading a model."""

import argparse
import json
from pathlib import Path

from backend.app import build_openapi


def write_openapi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_openapi(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_openapi(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
