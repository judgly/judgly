"""The external comparison's record reproduces: make compare-score's check (CPU only).

The frozen scorer, run on the committed answers, rebuilds every committed result file byte for
byte. It reads the tier files, which are not committed (make data builds them); without them the
test skips.
"""

import hashlib
import subprocess
import sys

import pytest

from conftest import REPO

RECORD = REPO / "docs" / "results" / "external-comparison"
TIERS = [REPO / "data" / "tiers" / f / f"{t}.jsonl"
         for f, t in [("general", "confirm"), ("stance", "confirm"), ("general", "final"),
                      ("stance", "final"), ("general", "bench"), ("general", "final-flagged"),
                      ("stance", "final-flagged")]]


def test_frozen_files_match_protocol_hashes():
    for line in (RECORD / "PROTOCOL.sha256").read_text().splitlines()[:3]:
        digest, name = line.split()
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == digest, name


def test_compare_score_rebuilds_committed_results(tmp_path):
    pytest.importorskip("numpy")
    missing = [str(p) for p in TIERS if not p.exists()]
    if missing:
        pytest.skip("tier files not built (make data): " + ", ".join(missing))
    proc = subprocess.run([sys.executable, str(RECORD / "reproduce.py"), "score", "--work", str(tmp_path)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all rebuilt results identical to the committed ones" in proc.stdout
    assert proc.stdout.count("identical to the committed file") == 8


def test_compare_run_refuses_to_write_into_the_record():
    proc = subprocess.run([sys.executable, str(RECORD / "reproduce.py"), "run", "--out", str(RECORD / "rerun")],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "never written over" in proc.stderr
    assert not (RECORD / "rerun").exists()
