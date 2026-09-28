"""Make variables for one model pack: the verified model file, the template and the engine
options the heads must be fitted under (the pack's own defaults, so that the heads match
what judgly.Engine.load(pack) runs).

    uv run --group pipeline python scripts/pack_info.py PACK [MODEL_DIR] [PREVIOUS] > pack-info.mk

The model file comes from MODEL_DIR when given (JUDGLY_MODEL_DIR), otherwise it is
downloaded from the pack's Hugging Face repository at the pinned revision; either way its
size and SHA-256 are checked against the pack. A MODEL_DIR that does not hold the pack's file
is an error, not a reason to download.

PREVIOUS is the pack-info.mk of an earlier (resumed) run of the same RESULTS directory. If
the model's SHA-256, the template's SHA-256 or an engine option differs from it, the script
fails: finished shards were made under the old settings. (The model's path may change.)
"""

import os
import sys
from pathlib import Path

from judgly.packs import Pack, resolve_model, sha256_file

FIXED = ("MODEL_SHA256", "TEMPLATE_SHA256", "ENGINE_ROTATIONS", "ENGINE_CONTENT_FREE",
         "ENGINE_MAX_ROTATIONS")


def main(pack: str, model_dir: str | None, previous: str | None) -> None:
    p = Pack.find(pack)
    if model_dir:
        if not (Path(model_dir) / p.model["filename"]).is_file():
            sys.exit(f"pack_info: MODEL_DIR={model_dir} does not hold {p.model['filename']}; "
                     "fix MODEL_DIR, or leave it out to download the file")
        os.environ["JUDGLY_MODEL_DIR"] = model_dir
    path, sha = resolve_model(p)
    opts = p.engine_options
    info = {"PACK_DIR": p.directory, "MODEL_PATH": path, "MODEL_SHA256": sha, "TEMPLATE": p.template,
            "TEMPLATE_SHA256": sha256_file(p.template),
            "ENGINE_ROTATIONS": 1 if opts.get("rotations", True) else 0,
            "ENGINE_CONTENT_FREE": 1 if opts.get("content_free", True) else 0,
            "ENGINE_MAX_ROTATIONS": int(opts.get("max_rotations", 0))}
    if previous and Path(previous).is_file():
        old = dict(line.split(" := ", 1) for line in Path(previous).read_text().splitlines() if " := " in line)
        changed = [k for k in FIXED if k in old and old[k] != str(info[k])]
        if changed:
            sys.exit(f"pack_info: {', '.join(changed)} changed since {previous} was written; the "
                     "finished shards used the old settings. Start a new RESULTS directory "
                     "(or delete this one) to run with the new ones.")
    for key, value in info.items():
        print(f"{key} := {value}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        sys.exit(__doc__)
    args = [a or None for a in sys.argv[2:]] + [None, None]
    main(sys.argv[1], args[0], args[1])
