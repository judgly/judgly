"""Download the `raw` files of data/registry.yaml into data/raw/<name>/ and check their SHA-256.

    uv run --group pipeline python scripts/fetch_raw.py

A file already present with the registered SHA-256 is kept. A download is written under a
temporary name and renamed only when its SHA-256 matches. Hugging Face sources are not fetched
here: the tier builder downloads them at their pinned revision into the Hugging Face cache.
Exit 1 if any file cannot be fetched or does not match.
"""

import sys
import urllib.request
from pathlib import Path

import registry


def files_of(name: str, raw: dict) -> dict[str, dict]:
    """Every file of one raw entry: {local file name: {url, sha256}}. The MultiVerS entry names
    one tarball; the others list their files."""
    if "files" in raw:
        return raw["files"]
    return {Path(raw["url"]).name: {"url": raw["url"], "sha256": raw["sha256"]}}


def fetch(path: Path, url: str, sha: str) -> str:
    if path.exists() and registry.sha256(path) == sha:
        return "present"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    got = registry.sha256(tmp)
    if got != sha:
        tmp.unlink()
        raise SystemExit(f"fetch_raw: {url}: SHA-256 {got}, registry says {sha}")
    tmp.rename(path)
    return "fetched"


def main() -> int:
    reg = registry.load()
    for name, raw in reg["raw"].items():
        for fname, f in files_of(name, raw).items():
            status = fetch(registry.RAW / name / fname, f["url"], f["sha256"])
            print(f"fetch_raw: {name}/{fname}: {status}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        sys.exit(__doc__)
    sys.exit(main())
