"""Check the documentation.

    JUDGLY_MODEL_DIR=/path/to/models uv run --group pipeline python docs/tools/check_docs.py

1. Every relative link and image in the Markdown files resolves to a file (and, for links into
   Markdown files with #anchors, to a heading). Absolute links into this repository at the
   release tag (github.com/judgly/judgly/{blob,tree}/vX, raw.githubusercontent.com) are checked
   the same way against the checkout, and the README (shown on PyPI) has no relative links.
2. Every examples/*.py runs to exit 0 against JUDGLY_PACK (default: the QUICK smoke pack
   results-quick/qwen3-4b-q8/pack), or is marked "# judgly-example: needs-full-pack" and
   compiles. The README quickstart runs too, with its pack swapped for JUDGLY_PACK.
3. CITATION.cff parses and has cff-version, title and authors.
4. docs/assets/social-preview.png is 1280 x 640; the SVGs parse as XML.
5. The source tables in docs/reproduce.md match data/registry.yaml.
"""

import os
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PACK = os.environ.get("JUDGLY_PACK", str(ROOT / "results-quick" / "qwen3-4b-q8" / "pack"))
OWN = re.compile(r"^https://(?:github\.com/judgly/judgly/(?:blob|tree)|"
                 r"raw\.githubusercontent\.com/judgly/judgly)/v[0-9][^/]*/(.*)$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|(?:src|href)=\"([^\"]+)\"")
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print("FAIL", msg)


def anchors(md: Path) -> set[str]:
    out = set()
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            out.add(re.sub(r"[^\w\- ]", "", title).replace(" ", "-"))
    return out


def markdown_files() -> list[Path]:
    files = [ROOT / n for n in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md",
                                "CODE_OF_CONDUCT.md", "SECURITY.md")]
    files += list((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def check_links() -> None:
    n = 0
    for md in markdown_files():
        text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
        for m in LINK.finditer(text):
            target = m.group(1) or m.group(2)
            own = OWN.match(target)
            if own:
                path, _, anchor = own.group(1).partition("#")
                dest = ROOT / path
            elif re.match(r"^[a-z]+:", target):
                continue
            else:
                if md.name == "README.md" and md.parent == ROOT:
                    fail(f"README.md: relative link {target} (PyPI cannot resolve it)")
                path, _, anchor = target.partition("#")
                dest = (md.parent / path).resolve() if path else md
            n += 1
            if not dest.exists():
                fail(f"{md.relative_to(ROOT)}: link to missing {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                fail(f"{md.relative_to(ROOT)}: no heading #{anchor} in {path or md.name}")
    print(f"links: {n} relative and own-repository links checked")


def run(cmd: list[str], label: str) -> None:
    env = {**os.environ, "JUDGLY_PACK": PACK}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        fail(f"{label} exited {r.returncode}: {r.stderr.strip().splitlines()[-1:]}")
    else:
        print(f"ok   {label}")


def check_examples() -> None:
    for ex in sorted((ROOT / "examples").glob("*.py")):
        src = ex.read_text(encoding="utf-8")
        if "# judgly-example: needs-full-pack" in src:
            compile(src, str(ex), "exec")
            print(f"ok   {ex.name} (compiles; marked as needing a full pack)")
        else:
            run([sys.executable, str(ex)], ex.name)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```python\n(.*?)```", readme, flags=re.S)
              if "engine.decide(" in b]
    for i, b in enumerate(blocks):
        code = b.replace('Engine.load("gemma4-12b-q8"', f"Engine.load({PACK!r}")
        run([sys.executable, "-c", code], f"README python block {i + 1}")


def check_files() -> None:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    for key in ("cff-version", "title", "authors", "message"):
        if key not in cff:
            fail(f"CITATION.cff lacks {key}")
    png = (ROOT / "docs" / "assets" / "social-preview.png").read_bytes()
    w, h = struct.unpack(">II", png[16:24])
    if (w, h) != (1280, 640):
        fail(f"social-preview.png is {w}x{h}, not 1280x640")
    for svg in (ROOT / "docs" / "assets").glob("*.svg"):
        ET.parse(svg)
    print("ok   CITATION.cff, social preview, SVGs")


def main() -> int:
    check_links()
    check_files()
    r = subprocess.run([sys.executable, str(ROOT / "docs" / "tools" / "sources_table.py"),
                        "--check"], cwd=ROOT)
    if r.returncode:
        fail("source tables out of date")
    check_examples()
    print("PASS docs" if not failures else f"FAIL docs: {len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
