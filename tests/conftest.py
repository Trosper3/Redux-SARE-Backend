"""
conftest.py — resolves engine imports for the Backend test suite.

The engines live in Backend/engines/ as a package with relative imports
(e.g. ``from .water_model import ...``).  The test files import them by
bare name (``from farm_selection_engine import ...``).

This conftest:
  1. Ensures Backend/ is on sys.path so ``import engines`` works.
  2. Pre-imports each engine through the package (which resolves all
     relative imports correctly).
  3. Registers each module under its bare name in sys.modules so that the
     bare-name imports in the test files resolve without an ImportError.
"""
import os
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from engines import water_model            # noqa: E402
from engines import water_network_engine   # noqa: E402
from engines import farm_selection_engine  # noqa: E402
from engines import water_scheduling_engine  # noqa: E402
from engines import min_st_cut_dual_engine   # noqa: E402

sys.modules.setdefault("water_model", water_model)
sys.modules.setdefault("water_network_engine", water_network_engine)
sys.modules.setdefault("farm_selection_engine", farm_selection_engine)
sys.modules.setdefault("water_scheduling_engine", water_scheduling_engine)
sys.modules.setdefault("min_st_cut_dual_engine", min_st_cut_dual_engine)
