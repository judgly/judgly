#!/bin/sh
cd "$(dirname "$0")"
echo "=== start $(date '+%Y-%m-%d %H:%M')" >> run.log
caffeinate -i uv run --no-project python run_calibration.py /Users/timo/code/judgly nimble:9b tev1:4b tev1:0.8b >> run.log 2>&1
echo "=== done $(date '+%Y-%m-%d %H:%M')" >> run.log
