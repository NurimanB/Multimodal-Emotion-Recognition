"""Package current tracked/new source for Colab before committing; omit data and weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/mer_experiments_colab.zip")
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    paths = []
    for name in sorted(set(listed) - {""}):
        path = ROOT / name
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        path.resolve().relative_to(ROOT)
        if set(Path(name).parts) & {".git", ".venv", ".venv311", "data", "outputs", "checkpoints", "dist", "__pycache__"}:
            # src/mer/data is source, not the root dataset directory.
            if not name.startswith("src/mer/data/"):
                continue
        if path.suffix in {".bin", ".pt", ".safetensors", ".joblib", ".pyc"}:
            continue
        paths.append((name, path))
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "includes_local_patch": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths},
    }
    if sum(path.stat().st_size for _, path in paths) > 20 * 1024**2:
        raise ValueError("Source bundle exceeds 20 MiB; inspect included files")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths:
            archive.write(path, name)
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    print(f"{output}: {len(paths)} source files, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
