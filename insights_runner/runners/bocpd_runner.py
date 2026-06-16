"""
BOCPDRunner — bridges RouterResult to insights_generation BOCPD pipeline.

Translation:
  RouterResult.classification.columns (subtype == "date")    → date column
  RouterResult.classification.columns (subtype == "measure") → target column
  RouterResult.column_profiles[target].skewness > 1.5        → apply log1p
  PARAMS["TRAIN_WEEKS"] = 180 (default)                      → train split
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult

from .base import ModelRunner

_TRAIN_WEEKS = 180
_HAZARD_LAM = 52
_CP_THRESHOLD = 0.30
_CP_MIN_DIST = 8


class BOCPDRunner(ModelRunner):
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        specs = router_result.classification.columns
        profiles = {p.name: p for p in router_result.column_profiles}

        date_col = next(
            (s.name for s in specs if s.semantic_subtype == "date"), None
        )
        measure_col = next(
            (s.name for s in specs if s.semantic_subtype == "measure"), None
        )

        if not date_col:
            return {"ran": False, "reason": "no date column identified"}
        if not measure_col:
            return {"ran": False, "reason": "no measure column identified"}
        if date_col not in df.columns:
            return {"ran": False, "reason": f"date column '{date_col}' missing from df"}
        if measure_col not in df.columns:
            return {"ran": False, "reason": f"measure column '{measure_col}' missing from df"}

        # ── Sort by date ─────────────────────────────────────────────────────
        try:
            mkt = df[[date_col, measure_col]].copy()
            mkt[date_col] = pd.to_datetime(mkt[date_col])
            mkt = mkt.sort_values(date_col).reset_index(drop=True)
        except Exception as exc:
            return {"ran": False, "reason": f"date parsing failed: {exc}"}

        # ── Log-transform target ──────────────────────────────────────────────
        target_profile = profiles.get(measure_col)
        skewness = target_profile.skewness if target_profile else None
        if skewness is not None and skewness > 1.5:
            log_series = np.log1p(mkt[measure_col].clip(lower=0).values.astype(float))
            transform_used = "log1p"
        else:
            log_series = np.log(mkt[measure_col].clip(lower=1e-6).values.astype(float))
            transform_used = "log"

        # ── Priors from training window ───────────────────────────────────────
        train_series = log_series[:_TRAIN_WEEKS]
        mu_0 = float(train_series.mean())
        diff_var = float(np.diff(train_series).var()) if len(train_series) > 1 else 0.0
        beta_0 = diff_var if diff_var > 0 else float(train_series.var())

        # ── Run BOCPD ─────────────────────────────────────────────────────────
        try:
            from pipeline.bocpd import extract_candidates, run_bocpd
        except ImportError:
            return {
                "ran": False,
                "reason": "bayesian_changepoint_detection / pipeline.bocpd not installed; "
                          "pip install -e insights_generation/",
            }

        try:
            out = run_bocpd(
                log_series,
                mu_0=mu_0,
                kappa_0=1.0,
                alpha_0=1.0,
                beta_0=beta_0,
                hazard_lam=_HAZARD_LAM,
            )
        except Exception as exc:
            return {"ran": False, "reason": f"run_bocpd failed: {exc}"}

        cp_prob = out["cp_prob"]
        exp_run_length = out["exp_run_length"]

        series_col_label = f"log_{measure_col}"
        try:
            candidates = extract_candidates(
                mkt[date_col],
                log_series,
                cp_prob,
                exp_run_length,
                threshold=_CP_THRESHOLD,
                min_dist=_CP_MIN_DIST,
                series_col=series_col_label,
            )
            cp_records = candidates.to_dict("records") if len(candidates) > 0 else []
            # Convert Timestamps to strings for JSON serialisation
            for rec in cp_records:
                for k, v in rec.items():
                    if isinstance(v, (pd.Timestamp,)):
                        rec[k] = str(v)
        except Exception as exc:
            cp_records = []

        cp_series = [
            {
                "week": str(mkt[date_col].iloc[i]),
                "cp_prob": float(cp_prob[i]),
                "exp_run_length": float(exp_run_length[i]),
            }
            for i in range(len(cp_prob))
        ]

        return {
            "ran": True,
            "signals": {
                "n_changepoints": len(cp_records),
                "cp_candidates": cp_records,
                "cp_probs_series": cp_series,
                "model_params": {
                    "mu_0": round(mu_0, 6),
                    "beta_0": round(beta_0, 8),
                    "hazard_lam": _HAZARD_LAM,
                    "threshold": _CP_THRESHOLD,
                    "min_dist": _CP_MIN_DIST,
                    "target_transform": transform_used,
                },
            },
        }
