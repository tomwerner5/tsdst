"""Fail if a package function has no executed body lines in a coverage report.

Run after pytest with coverage:
    python scripts/check_function_coverage.py coverage.json
"""

import json
import sys
from pathlib import Path


def main():
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    missing = []
    count = 0
    for filename, data in report["files"].items():
        if "functions" not in data:
            raise SystemExit("Use coverage 7.10 or newer to report function coverage")
        for name, function in data["functions"].items():
            if not name:
                continue
            count += 1
            if not function["executed_lines"]:
                missing.append("{}:{}: {}".format(filename, function["start_line"], name))
    if missing:
        raise SystemExit("Functions without test execution:\n" + "\n".join(missing))
    print("Executed all {} package functions and methods.".format(count))


if __name__ == "__main__":
    main()
