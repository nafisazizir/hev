"""Checkpoint references: a local run directory or an hf://OWNER/name[@revision] Hub snapshot."""
from pathlib import Path

HUB_PREFIX = "hf://"


def parse_hub_reference(reference):
    """Split an hf://OWNER/name[@revision] reference into (repo_id, revision)."""
    text = str(reference)
    if not text.startswith(HUB_PREFIX):
        raise ValueError(f"not a Hub reference (missing {HUB_PREFIX}): {reference}")
    body = text[len(HUB_PREFIX):]
    repo_id, sep, revision = body.partition("@")
    if sep and not revision:
        raise ValueError(f"empty revision in Hub reference: {reference}")
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Hub repo must be exactly owner/name: {repo_id}")
    return repo_id, revision or None


def resolve_run(reference):
    """Return a local path for a run: existing paths win, hf:// refs download a snapshot."""
    path = Path(reference).expanduser()
    if path.exists():
        return path
    if not str(reference).startswith(HUB_PREFIX):
        raise FileNotFoundError(f"run not found: {reference}")
    repo_id, revision = parse_hub_reference(reference)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, revision=revision))
