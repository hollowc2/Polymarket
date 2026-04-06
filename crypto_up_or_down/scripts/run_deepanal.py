#!/usr/bin/env python3
"""Launch the deepanal strategy explorer.

Usage:
    uv run python scripts/run_deepanal.py
    uv run python scripts/run_deepanal.py --port 8502
"""

import subprocess
import sys
from pathlib import Path

app = Path(__file__).parent / "deepanal" / "app.py"

cmd = ["streamlit", "run", str(app), "--server.headless", "true"]

# Pass through any extra args (e.g. --port 8502) after a bare --
extra = sys.argv[1:]
if extra:
    cmd += ["--"] + extra

subprocess.run(cmd)
