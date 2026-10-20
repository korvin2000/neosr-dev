import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# TEMPORARY HACK: `import neosr` parses sys.argv at import time (neosr/__init__ ->
# data -> utils.rng -> parse_options). Point it at a minimal TOML BEFORE any neosr
# import. Remove together with the import-time argv parsing (G5.8).
sys.argv = [sys.argv[0], "-opt", str(Path(__file__).with_name("loss_test_options.toml"))]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
