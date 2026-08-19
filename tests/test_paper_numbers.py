"""the paper's number checker has to run, and has to pass.

`tools/check_paper_numbers.py` is advertised in the paper's Artifact
section as the script that checks every number against the committed
records. It shipped once with a NameError that made it unrunnable
(ERRATA 2.19), which nothing caught because nothing ran it. It runs here
now: a checker no suite executes is not a checker.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_checker_runs_and_reports_no_failures():
    p = subprocess.run([sys.executable, "tools/check_paper_numbers.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (
        f"checker exited {p.returncode}\n"
        f"--- stdout tail ---\n{p.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{p.stderr[-2000:]}")
    assert "0 FAIL" in p.stdout, p.stdout[-2000:]
