#!/bin/sh
# clef:27b on the test tiers, then the calibration sample. Resumable: rerun this script after an interruption.
cd "$(dirname "$0")"
echo "=== clef:27b test start $(date '+%Y-%m-%d %H:%M')" >> run.log
caffeinate -i uv run --no-project python run_external.py /Users/timo/code/judgly clef:27b >> run.log 2>&1
echo "=== clef:27b test done $(date '+%Y-%m-%d %H:%M')" >> run.log
echo "=== clef:27b train start $(date '+%Y-%m-%d %H:%M')" >> run.log
caffeinate -i uv run --no-project python calibrated/run_calibration.py /Users/timo/code/judgly clef:27b >> run.log 2>&1
echo "=== clef:27b train done $(date '+%Y-%m-%d %H:%M')" >> run.log
