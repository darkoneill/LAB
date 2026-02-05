#!/usr/bin/env python3
"""
OpenClaw - Launch Script
Usage:
    python run.py              # Run terminal + gateway (default)
    python run.py terminal     # Terminal only
    python run.py gateway      # Gateway API only
    python run.py wizard       # Run setup wizard
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent))

from openclaw.main import main

if __name__ == "__main__":
    main()
