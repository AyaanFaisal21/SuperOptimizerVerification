"""puts the repo root on sys.path so `import fpgap` works without an install step.

deliberately not packaging this: the repo is a measurement artifact meant to be
cloned and run, and `pip install -e .` is one more thing between a reader and a
reproduction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
