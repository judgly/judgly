"""Assert that an s1-eval JSON report holds every metric of the evaluation report with a
finite value. The values themselves are results to be read, not part of the check.

    uv run --group pipeline python scripts/check_report.py REPORT.json [REPORT.json ...]
"""

import json
import math
import sys

METRICS = ["accuracy", "log_loss", "brier", "ece", "slot_mass", "slot_mass_p05"]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def problems(report: dict) -> list[str]:
    found = []
    for name in METRICS:
        entry = report["metrics"].get(name, {})
        found += [f"{name}.{field}" for field in ("value", "lo", "hi") if not finite(entry.get(field))]
    if len(report["reliability"]) != 10 or sum(b["n"] for b in report["reliability"]) != report["items"]:
        found.append("reliability table does not cover the items")
    for b in report["reliability"]:
        found += [f"reliability bin {b['bin']}" for k in ("confidence", "accuracy", "wilson_lo", "wilson_hi")
                  if b["n"] > 0 and not finite(b[k])]
    if not report["risk_coverage"] or not all(finite(r["coverage"]) for r in report["risk_coverage"]):
        found.append("risk and coverage")
    return found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    failed = False
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        bad = problems(report)
        m = report["metrics"]
        print(f"{'FAIL' if bad else 'ok  '} {report['condition']:45s} n={report['items']:4d}  "
              f"accuracy {m['accuracy']['value']:.3f}  log loss {m['log_loss']['value']:.3f}  "
              f"ECE {m['ece']['value']:.3f}" + (f"  problems: {bad}" if bad else ""))
        failed |= bool(bad)
    sys.exit(1 if failed else 0)
