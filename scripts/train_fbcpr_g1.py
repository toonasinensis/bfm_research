from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_fbcpr_humenv import main


if __name__ == "__main__":
    if not any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv):
        sys.argv.extend(["--task", "g1"])
    main()
