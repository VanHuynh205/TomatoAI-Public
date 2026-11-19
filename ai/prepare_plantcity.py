"""Pack the six mapped classes from the verified PlantCity v2 original archive."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo

SOURCE = "https://data.mendeley.com/datasets/w8kh2xkspx/2"
DOWNLOAD = (
    "https://data.mendeley.com/public-files/datasets/w8kh2xkspx/files/"
    "cc5d2e50-03cd-450c-a444-4c6f48683d45/file_downloaded"
)
ARCHIVE_SHA256 = "db1bd5624f42aceebff4d8ff08f99257b0608ac712a8bf5bf7f1e676d68bfffa"
CLASSES = {
    "tomato_bacterial_spot": ("bacterial_spot", 400),
    "tomato_early_blight": ("early_blight", 400),
    "tomato_late_blight": ("late_blight", 294),
    "tomato_leaf_mold": ("leaf_mold", 400),
    "tomato_septoria_leaf": ("septoria_leaf_spot", 341),
    "tomato_healthy_leaf": ("healthy", 400),
}


def prepare(source: Path, output: Path) -> Counter[str]:
    if source.resolve() == output.resolve():
        raise ValueError("The output must differ from the source archive")
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != ARCHIVE_SHA256:
            raise ValueError(f"Unexpected SHA-256 for PlantCity v2 test.zip: {actual}")
        stream.seek(0)
        with ZipFile(stream) as original, TemporaryDirectory(dir=output.parent) as temp:
            staged = Path(temp) / output.name
            counts: Counter[str] = Counter()
            samples: list[dict[str, str]] = []
            paths = set()
            with ZipFile(staged, "w") as subset:
                for member in sorted(
                    original.infolist(), key=lambda item: item.filename
                ):
                    parts = PurePosixPath(member.filename).parts
                    if (
                        member.is_dir()
                        or len(parts) != 3
                        or parts[0] != "test"
                        or parts[1] not in CLASSES
                        or PurePosixPath(parts[2]).suffix.lower() != ".jpg"
                    ):
                        continue
                    class_id, _ = CLASSES[parts[1]]
                    name = f"{class_id}/{parts[2]}"
                    if name.casefold() in paths:
                        raise ValueError(f"Duplicate destination path: {name}")
                    paths.add(name.casefold())
                    payload = original.read(member)
                    subset.writestr(ZipInfo(name), payload)
                    counts[class_id] += 1
                    samples.append(
                        {
                            "path": name,
                            "source_member": member.filename,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                expected = {class_id: count for class_id, count in CLASSES.values()}
                if counts != expected:
                    raise ValueError(
                        f"Unexpected original image counts: {dict(counts)}"
                    )
                receipt = {
                    "dataset": "PlantCity v2",
                    "source": SOURCE,
                    "license": "CC BY 4.0",
                    "archive": "test.zip",
                    "archive_url": DOWNLOAD,
                    "archive_sha256": actual,
                    "role": "train/validation candidate only",
                    "counts": dict(counts),
                    "samples": samples,
                }
                subset.writestr(
                    ZipInfo("receipt.json"), json.dumps(receipt, indent=2) + "\n"
                )
            staged.replace(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified PlantCity v2 test.zip")
    parser.add_argument("output", type=Path, help="Destination subset ZIP")
    args = parser.parse_args()
    print(dict(prepare(args.source, args.output)))
