"""
RUNNER_REGISTRY maps model_name → ModelRunner subclass.

To add a new model:
  1. Create insights_runner/runners/<model>_runner.py implementing ModelRunner
  2. Import it here and add to RUNNER_REGISTRY
  No other files need to change.
"""

from .bocpd_runner import BOCPDRunner
from .mmm_runner import MMMRunner
from .psi_runner import PSIRunner
from .arima_runner import ARIMARunner

RUNNER_REGISTRY: dict = {
    "BOCPD": BOCPDRunner,
    "MMM": MMMRunner,
    "PSI": PSIRunner,
    "ARIMA": ARIMARunner,
}

__all__ = ["RUNNER_REGISTRY"]
