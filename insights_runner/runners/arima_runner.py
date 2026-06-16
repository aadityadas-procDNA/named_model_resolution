"""ARIMARunner — stub. Quality gate runs; pipeline not yet implemented."""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult

from .base import ModelRunner


class ARIMARunner(ModelRunner):
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        return {
            "ran": False,
            "reason": "ARIMA pipeline not yet implemented — quality gate only",
        }
