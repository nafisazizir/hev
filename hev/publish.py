"""Publish one evaluated run to a single Hub model repo, one branch per seed.

    uv run python -m hev.publish --run runs/m2-pointer-s0 --repo OWNER/hev-0.6b --revision seed-0 --dry-run
"""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch

REQUIRED = ["adapter_config.json", "adapter_model.safetensors", "readout.pt",
            "result.json", "training_config.json", "training_metrics.json"]
OPTIONAL = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "added_tokens.json", "special_tokens_map.json"]
SEED_REVISION = re.compile(r"^seed-(\d+)$")
CARD_PLACEHOLDER = "OWNER/hev-0.6b"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD = ROOT / "MODEL_CARD.md"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(run, revision):
    """Check a run directory is a servable, fully-evaluated pointer checkpoint; return its identity."""
    run = Path(run)
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run is missing required files: {missing}")
    training_metrics = json.loads((run / "training_metrics.json").read_text())
    if training_metrics.get("status") != "success":
        raise ValueError("checkpoint training did not complete successfully")
    result = json.loads((run / "result.json").read_text())
    if result.get("status") != "success":
        raise ValueError("run evaluation did not complete successfully")
    if result.get("test_evaluated") is not False:
        raise ValueError("refusing to publish a checkpoint evaluated on the locked test split")
    metadata = torch.load(run / "readout.pt", map_location="cpu", weights_only=True)
    if result.get("head") != "pointer" or metadata["readout"]["head_kind"] != "pointer":
        raise ValueError("only pointer-head checkpoints are publishable")
    identity = {
        "base": metadata["base"],
        "base_revision": metadata["base_revision"],
        "training_suite_sha256": metadata["suite_sha256"],
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("result provenance does not match the checkpoint")
    training_config = json.loads((run / "training_config.json").read_text())
    if training_config.get("status") != "configured":
        raise ValueError("checkpoint training config is not marked configured")
    if training_config != result.get("training_config"):
        raise ValueError("result training config does not match the checkpoint")
    if training_config.get("config_sha256") != metadata.get("training_config_sha256"):
        raise ValueError("result training config does not match the checkpoint")
    match = SEED_REVISION.fullmatch(revision)
    if not match:
        raise ValueError("revision must be seed-N")
    seed = training_config.get("args", {}).get("seed")
    if not isinstance(seed, int) or metadata.get("args", {}).get("seed") != seed:
        raise ValueError("checkpoint seed provenance does not match")
    if int(match.group(1)) != seed:
        raise ValueError(f"revision {revision} does not match recorded training seed {seed}")
    return {"run": run.name, "seed": seed, "metadata": metadata, "result": result}


def build_bundle(run, repo, revision, card=None):
    """Validate a run and stage an upload folder; returns (TemporaryDirectory, manifest)."""
    run = Path(run)
    info = validate_run(run, revision)
    tmp = tempfile.TemporaryDirectory(prefix="hev-publish-")
    dest = Path(tmp.name)
    names = [n for n in REQUIRED + OPTIONAL if (run / n).is_file()]
    files = {}
    for name in names:
        shutil.copy2(run / name, dest / name)
        files[name] = {"sha256": _sha256(run / name), "bytes": (run / name).stat().st_size}
    card = Path(card) if card else DEFAULT_CARD
    if not card.is_file():
        raise FileNotFoundError(f"model card not found: {card}")
    (dest / "README.md").write_text(card.read_text().replace(CARD_PLACEHOLDER, repo))
    manifest = {
        "schema_version": 1,
        "repo": repo,
        "revision": revision,
        "run": info["run"],
        "base": info["metadata"]["base"],
        "base_revision": info["metadata"]["base_revision"],
        "head": "pointer",
        "seed": info["seed"],
        "test_evaluated": False,
        "files": files,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return tmp, manifest


def main():
    parser = argparse.ArgumentParser(description="publish one evaluated hev run to the Hugging Face Hub")
    parser.add_argument("--run", required=True, help="local run directory")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. OWNER/hev-0.6b")
    parser.add_argument("--revision", required=True, help="branch to upload to, e.g. seed-0")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default=None, help="commit message")
    parser.add_argument("--card", default=None,
                        help="model card markdown; defaults to the repository MODEL_CARD.md")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the bundle; no network")
    args = parser.parse_args()

    tmp, manifest = build_bundle(args.run, args.repo, args.revision, card=args.card)
    with tmp:
        if args.dry_run:
            files = sorted(p.name for p in Path(tmp.name).iterdir())
            print(json.dumps({"repo": args.repo, "revision": args.revision,
                              "run": manifest["run"], "files": files}))
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
        readme = Path(tmp.name) / "README.md"
        if readme.is_file():
            api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                            repo_id=args.repo, repo_type="model", revision="main",
                            commit_message=args.message or "update model card")
        api.create_branch(args.repo, repo_type="model", branch=args.revision, exist_ok=True)
        api.upload_folder(folder_path=str(tmp.name), repo_id=args.repo, repo_type="model",
                          revision=args.revision,
                          commit_message=args.message or f"publish {manifest['run']}")
        print(f"https://huggingface.co/{args.repo}/tree/{args.revision}")


if __name__ == "__main__":
    main()
