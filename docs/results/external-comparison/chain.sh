#!/bin/sh
cd /private/tmp/claude-502/-Users-timo-code-sysone/f2abc4b5-e64e-425e-9f4e-8aeafe12e5aa/scratchpad/external
for m in tev1:4b tev1:0.8b nimble:9b; do
  until ollama list | grep -q "^$m "; do sleep 30; done
  echo "=== $m start $(date '+%H:%M')" >> run.log
  caffeinate -i uv run --no-project python run_external.py /Users/timo/code/judgly $m >> run.log 2>&1
  echo "=== $m done $(date '+%H:%M')" >> run.log
done
echo "=== all done $(date '+%H:%M')" >> run.log
