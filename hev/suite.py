"""Frozen evaluation suites (evals/<name>/): checksummed jsonl splits plus a manifest.

Suites are copied byte-for-byte from kev (docs/EVALS.md lists provenance). Rules:
  - load_split verifies sha256 and record counts against the manifest; a mismatch is an error, never a warning.
  - the `test` split is locked: it is refused unless allow_test=True, and must never be used for model selection.
  - training must refuse any record whose source is eval-only (hev.data.EVAL_ONLY).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "calibration", "development", "test")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes(root=ROOT):
    root = Path(root)
    paths = list((root / "hev").glob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths) if path.exists()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text())


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test split requires allow_test=True; never use it for model selection")
    directory = Path(directory)
    m = manifest(directory)
    path = directory / f"{split}.jsonl"
    if digest(path) != m["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != m["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def base_revision(directory, base):
    """Pinned Hub commit for a base model, or None if the suite does not pin it."""
    return manifest(directory).get("base_revisions", {}).get(base)
