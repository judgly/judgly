"""Join the feature files of extraction shards into one feature file.

    uv run --group pipeline python scripts/merge_features.py OUT.feat SHARD.feat [SHARD.feat ...]

A feature file (csrc/s1_feat.c) is a 200-byte header followed by fixed-size records, so shards
extracted separately join by concatenating their records and writing the summed count into the
header. The shards must agree on every header byte except the count (same model file, template,
width, flags and slot ids), and each must be complete: s1-features writes the count only when it
closes the file, so a shard whose count is 0 or whose size does not match its count was
interrupted. Records keep the order of the shards on the command line. The output is written
to OUT.feat.tmp and renamed, so an interrupted merge leaves no OUT.feat behind.
"""

import os
import struct
import sys

HEADER_BYTES = 96 + 4 * 26  # csrc/s1_feat.c: header and the 26 slot token ids
RECORD_BYTES = 8 + 4 + 2 + 5 + 26 + 4 * (3 * 26 + 1)  # without the n_embd floats of h
COUNT_OFFSET = 24
MAGIC = b"S1FEAT\0\0"


def read_header(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
    if len(header) != HEADER_BYTES or header[:8] != MAGIC:
        sys.exit(f"merge_features: {path} is not a feature file")
    n_embd = struct.unpack_from("<I", header, 12)[0]
    count = struct.unpack_from("<Q", header, COUNT_OFFSET)[0]
    expected = HEADER_BYTES + count * (RECORD_BYTES + 4 * n_embd)
    if count == 0 or os.path.getsize(path) != expected:
        sys.exit(f"merge_features: {path} is incomplete ({count} records, "
                 f"{os.path.getsize(path)} bytes, {expected} expected); delete it and extract again")
    return header, count


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, shards = sys.argv[1], sys.argv[2:]
    first, total = read_header(shards[0])
    for path in shards[1:]:
        header, count = read_header(path)
        if header[:COUNT_OFFSET] + header[COUNT_OFFSET + 8:] != first[:COUNT_OFFSET] + first[COUNT_OFFSET + 8:]:
            sys.exit(f"merge_features: {path} differs from {shards[0]} in model, template, width, "
                     "flags or slot ids")
        total += count
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(first[:COUNT_OFFSET] + struct.pack("<Q", total) + first[COUNT_OFFSET + 8:])
        for path in shards:
            with open(path, "rb") as shard:
                shard.seek(HEADER_BYTES)
                while block := shard.read(1 << 24):
                    f.write(block)
    os.replace(tmp, out)
    print(f"merge_features: {len(shards)} shards, {total} records -> {out}")


if __name__ == "__main__":
    main()
