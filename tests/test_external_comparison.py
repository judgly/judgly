"""The external comparison's record reproduces: make compare-score's check (CPU only).

The frozen scorer, run on the committed answers, rebuilds every committed result file byte for
byte; so does the equal-calibration control's frozen scorer (calibrated/). Both read the tier
files, which are not committed (make data builds them); without them the tests skip. The control
needs scipy, which the project environment does not have: its test runs the pinned command of
make compare-score through uv, and skips without uv.
"""

import hashlib
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO

RECORD = REPO / "docs" / "results" / "external-comparison"
CONTROL = RECORD / "calibrated"
TIERS = [REPO / "data" / "tiers" / f / f"{t}.jsonl"
         for f, t in [("general", "confirm"), ("stance", "confirm"), ("general", "final"),
                      ("stance", "final"), ("general", "bench"), ("general", "final-flagged"),
                      ("stance", "final-flagged")]]
FITDEV = [REPO / "data" / "tiers" / f / "fitdev.jsonl" for f in ("general", "stance")]


def test_frozen_files_match_protocol_hashes():
    for line in (RECORD / "PROTOCOL.sha256").read_text().splitlines()[:3]:
        digest, name = line.split()
        assert hashlib.sha256((RECORD / name).read_bytes()).hexdigest() == digest, name


def test_compare_score_rebuilds_committed_results(tmp_path):
    pytest.importorskip("numpy")
    missing = [str(p) for p in TIERS if not p.exists()]
    if missing:
        pytest.skip("tier files not built (make data): " + ", ".join(missing))
    proc = subprocess.run([sys.executable, str(RECORD / "reproduce.py"), "score", "--part", "comparison",
                           "--work", str(tmp_path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all rebuilt results identical to the committed ones" in proc.stdout
    assert proc.stdout.count("identical to the committed file") == 8


def test_control_frozen_files_match_protocol_hashes():
    lines = [x for x in (CONTROL / "PROTOCOL.sha256").read_text().splitlines()
             if re.match(r"^[0-9a-f]{64}  \S+$", x)]
    assert len(lines) == 5
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((CONTROL / name).read_bytes()).hexdigest() == digest, name


def test_control_rescore_rebuilds_committed_results(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH (the control's scorer runs with the pinned numpy and scipy)")
    missing = [str(p) for p in TIERS + FITDEV if not p.exists()]
    if missing:
        pytest.skip("tier files not built (make data): " + ", ".join(missing))
    proc = subprocess.run([uv, "run", "--no-project", "--python", "3.13", "--with", "numpy==2.5.3",
                           "--with", "scipy==1.18.1", "python", str(RECORD / "reproduce.py"), "score",
                           "--part", "control", "--work", str(tmp_path)],
                          capture_output=True, text=True, check=False, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all rebuilt results identical to the committed ones" in proc.stdout
    assert proc.stdout.count("identical to the committed file") == 2


def test_compare_run_refuses_to_write_into_the_record():
    proc = subprocess.run([sys.executable, str(RECORD / "reproduce.py"), "run", "--out", str(RECORD / "rerun")],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "never written over" in proc.stderr
    assert not (RECORD / "rerun").exists()
