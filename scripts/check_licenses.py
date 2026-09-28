"""Fail unless every vendored component has its licence text in LICENSES/ and an entry in NOTICE.

Run from the repository root: uv run python scripts/check_licenses.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Each component compiled into libjudgly or the tools: its licence file in LICENSES/, the
# upstream file it must match byte for byte (None when assembled from a source header), and a
# string that must appear in NOTICE.
COMPONENTS = {
    "llama.cpp and ggml": ("llama.cpp-ggml-MIT.txt", "third_party/llama.cpp/LICENSE", "llama.cpp and ggml"),
    "llamafile sgemm": ("llamafile-sgemm-MIT.txt", None, "llamafile sgemm"),
    "YaRN RoPE": ("yarn-rope-MIT.txt", None, "YaRN RoPE"),
    "yyjson": ("yyjson-MIT.txt", "third_party/yyjson/LICENSE", "yyjson"),
    "ggllm.cpp tokenizer": ("ggllm-cpp-MIT.txt", None, "ggllm.cpp tokenizer"),
}

# Every directory under third_party/ must be covered by a component above.
VENDORED_DIRS = {"llama.cpp": "llama.cpp and ggml", "yyjson": "yyjson"}

# Source files with their own licence notice, and the component that covers each.
MARKED_SOURCES = {
    "third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp": ("Copyright 2024 Mozilla Foundation", "llamafile sgemm"),
    "third_party/llama.cpp/ggml/src/ggml-cpu/ops.cpp": ("Jeffrey Quesnelle and Bowen Peng", "YaRN RoPE"),
    "third_party/llama.cpp/ggml/src/ggml-metal/kernels/rope.metal": ("Jeffrey Quesnelle and Bowen Peng", "YaRN RoPE"),
    "third_party/llama.cpp/src/llama-vocab.cpp": ("cmp-nct/ggllm.cpp", "ggllm.cpp tokenizer"),
}


def main() -> int:
    errors = []
    notice = (ROOT / "NOTICE").read_text()

    for name, (licence, upstream, notice_key) in COMPONENTS.items():
        path = ROOT / "LICENSES" / licence
        if not path.is_file() or not path.read_text().strip():
            errors.append(f"{name}: LICENSES/{licence} missing or empty")
            continue
        if upstream is not None:
            source = ROOT / upstream
            if not source.is_file():
                errors.append(f"{name}: upstream licence {upstream} missing (submodule not checked out?)")
            elif source.read_bytes() != path.read_bytes():
                errors.append(f"{name}: LICENSES/{licence} differs from {upstream}")
        if notice_key not in notice:
            errors.append(f"{name}: no entry in NOTICE")

    for entry in sorted((ROOT / "third_party").iterdir()):
        if entry.is_dir() and entry.name not in VENDORED_DIRS:
            errors.append(f"third_party/{entry.name}: vendored but has no licence entry in this script")

    for rel, (marker, component) in MARKED_SOURCES.items():
        source = ROOT / rel
        if source.is_file() and marker not in source.read_text(errors="replace"):
            errors.append(f"{rel}: expected notice '{marker}' ({component}) not found; recheck licences")

    for e in errors:
        print(f"check_licenses: {e}", file=sys.stderr)
    if not errors:
        print(f"check_licenses: {len(COMPONENTS)} components, all licences present")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
