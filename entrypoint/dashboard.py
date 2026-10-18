#!/usr/bin/env python3
"""
Dashboard launcher entrypoint.
Usage:
    streamlit run entrypoint/dashboard.py
"""

import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DASHBOARD_FILE = ROOT / "dashboard" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(DASHBOARD_FILE), run_name="__main__")
