"""Abstract base class for model runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult


class ModelRunner(ABC):
    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        """
        Execute the model pipeline and return a signals dict.

        The returned dict MUST have a top-level "ran" key (bool).
        If ran=True, signals live under a "signals" key.
        If ran=False, a "reason" key explains why.
        """
