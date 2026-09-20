"""Frozen evaluation suites (evals/<name>/): checksummed jsonl splits plus a manifest.

Suites are copied byte-for-byte from kev (docs/EVALS.md lists provenance). Rules:
  - load_split verifies sha256 and record counts against the manifest; a mismatch is an error, never a warning.
  - the `test` split is locked: it is refused unless allow_test=True, and must never be used for model selection.
  - training must refuse any record whose source the suite does not declare trainable (hev.data.source_policy).
  - a partition too large for git is fetched from kev's pinned Hub mirror, but only when the caller passes
    fetch=True, so reading a suite never reaches the network by accident; the bytes are verified either way.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "calibration", "development", "test")

# kev does not keep training partitions over 10 MB in git; it mirrors every frozen suite on this Hub dataset
# and fetches a missing partition on first use (kev/suite.py::fetch_partition at 20fa626). Hev does the same,
# at the same pinned revision, so `decision-v4/train.jsonl` reaches disk with the bytes the manifest names and
# never enters this repository. Hev's suites live flat in evals/ while the mirror keeps kev's nested layout,
# so the mapping is written out rather than derived.
SUITES_DATASET = "jaredpalmer/kev-suites"
SUITES_REVISION = "a3318ddc1f630c5673232efacd8123a84de3f480"
MIRROR_PATHS = {"decision-v4": "v4/decision-v4", "transfer-v4": "v4/transfer-v4"}


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


def mirror_path(directory, filename):
    """Path of one partition inside the Hub mirror, or an error if this suite has no mirrored copy."""
    name = Path(directory).resolve().name
    if name not in MIRROR_PATHS:
        raise FileNotFoundError(f"{Path(directory) / filename} is missing and {name} has no Hub mirror entry")
    return f"{MIRROR_PATHS[name]}/{filename}"


def fetch_partition(directory, filename):
    """Download one partition from the pinned mirror into place, refusing bytes the manifest does not name."""
    import shutil
    from huggingface_hub import hf_hub_download
    directory = Path(directory)
    relative = mirror_path(directory, filename)
    expected = manifest(directory)["files"][filename]["sha256"]
    cached = hf_hub_download(SUITES_DATASET, relative, repo_type="dataset", revision=SUITES_REVISION)
    if digest(cached) != expected:
        raise ValueError(f"mirrored partition does not match the manifest: {relative}")
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {SUITES_DATASET}@{SUITES_REVISION[:10]}", flush=True)
    return directory / filename


def load_split(directory, split, allow_test=False, fetch=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test split requires allow_test=True; never use it for model selection")
    directory = Path(directory)
    m = manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists() and fetch and split != "test":
        fetch_partition(directory, path.name)   # opt-in: the offline read path never touches the network
    if digest(path) != m["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != m["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def base_revision(directory, base):
    """Pinned Hub commit for a base model, or None if the suite does not pin it."""
    return manifest(directory).get("base_revisions", {}).get(base)
