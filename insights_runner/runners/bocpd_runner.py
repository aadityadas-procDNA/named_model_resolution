"""
BOCPDRunner — bridges RouterResult to pipeline BOCPD implementation.

Translation:
  RouterResult.classification.columns (subtype == "date")    -> date column
  RouterResult.classification.columns (subtype == "measure") -> target column
  RouterResult.column_profiles[target].skewness > 1.5        -> apply log1p
  RouterResult.column_profiles[date].date_grain              -> hazard_lam calibration
  _TRAIN_PERIODS = 180                                        -> train split (row count)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult

from .base import ModelRunner

_TRAIN_PERIODS = 180       # row count for training window (grain-agnostic)
_HAZARD_LAM   = 52         # default expected run length (periods); overridden by grain
_CP_THRESHOLD = 0.30
_CP_MIN_DIST  = 8          # minimum periods between changepoints
_CP_WINDOW    = 8          # half-width of context window around each changepoint (periods)

# Periods per year by grain — used to calibrate hazard_lam
_PERIODS_PER_YEAR: dict[str, int] = {
    "daily":   365,
    "weekly":   52,
    "monthly":  12,
}


class BOCPDRunner(ModelRunner):
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        specs    = router_result.classification.columns
        profiles = {p.name: p for p in router_result.column_profiles}

        date_col = next(
            (s.name for s in specs if s.semantic_subtype == "date"), None
        )

        from ._measure_selector import select_measure_column
        measure_col, measure_note = select_measure_column(specs, profiles)

        if not date_col:
            return {"ran": False, "reason": "no date column identified"}
        if not measure_col:
            return {"ran": False, "reason": "no measure column identified — no measure or unclassified_metric columns found"}
        if date_col not in df.columns:
            return {"ran": False, "reason": f"date column '{date_col}' missing from df"}
        if measure_col not in df.columns:
            return {"ran": False, "reason": f"measure column '{measure_col}' missing from df"}

        # ── Sort by date (with geographic rollup if needed) ──────────────────
        try:
            mkt = df[[date_col, measure_col]].copy()
            mkt[date_col] = pd.to_datetime(mkt[date_col])

            # Collapse territory/HCP rows to a single national time series.
            # BOCPD requires one observation per time step.
            from ._data_normalizer import normalize_to_series
            mkt, agg_note = normalize_to_series(mkt, date_col, [measure_col])
            if agg_note:
                measure_note = (
                    (measure_note + " " + agg_note).strip() if measure_note else agg_note
                )

            mkt = mkt.sort_values(date_col).reset_index(drop=True)
        except Exception as exc:
            return {"ran": False, "reason": f"date parsing failed: {exc}"}

        # ── Detect date grain and calibrate hazard_lam ────────────────────────
        # hazard_lam = expected run length in periods (how many periods between CPs).
        # Calibrating to ~1 year of periods makes the prior scale-invariant across grains.
        date_profile = profiles.get(date_col)
        grain = date_profile.date_grain if date_profile else None
        hazard_lam = _PERIODS_PER_YEAR.get(grain, _HAZARD_LAM)

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
        train_series = log_series[:_TRAIN_PERIODS]
        mu_0    = float(train_series.mean())
        diff_var = float(np.diff(train_series).var()) if len(train_series) > 1 else 0.0
        beta_0  = diff_var if diff_var > 0 else float(train_series.var())

        # ── Run BOCPD ─────────────────────────────────────────────────────────
        try:
            from pipeline.bocpd import extract_candidates, run_bocpd
        except ImportError:
            return {
                "ran": False,
                "reason": "bayesian_changepoint_detection / pipeline.bocpd not installed; "
                          "pip install -e .[pipeline]",
            }

        try:
            out = run_bocpd(
                log_series,
                mu_0=mu_0,
                kappa_0=1.0,
                alpha_0=1.0,
                beta_0=beta_0,
                hazard_lam=hazard_lam,
            )
        except Exception as exc:
            return {"ran": False, "reason": f"run_bocpd failed: {exc}"}

        cp_prob        = out["cp_prob"]
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
                    if isinstance(v, pd.Timestamp):
                        rec[k] = str(v)
        except Exception:
            cp_records = []

        # ── Context windows around each changepoint ───────────────────────────
        # Keys are named after the actual measure column so they are meaningful
        # regardless of grain or metric type (TRX, NBRX, adherence_rate, …).
        log_col_key = f"log_{measure_col}"
        raw_col_key = measure_col
        cp_context_windows = []
        for rec in cp_records:
            idx   = int(rec["week_idx"])
            start = max(0, idx - _CP_WINDOW)
            end   = min(len(log_series), idx + _CP_WINDOW + 1)

            window_series = [
                {
                    "date":           str(mkt[date_col].iloc[i]),
                    log_col_key:      round(float(log_series[i]), 4),
                    raw_col_key:      round(float(mkt[measure_col].iloc[i]), 2),
                    "cp_prob":        round(float(cp_prob[i]), 4),
                    "exp_run_length": round(float(exp_run_length[i]), 2),
                }
                for i in range(start, end)
            ]

            cp_context_windows.append({
                "changepoint_date":      rec["week"],
                "cp_prob":               rec["cp_prob"],
                "exp_run_length":        rec["exp_run_length"],
                f"{log_col_key}_at_cp":  rec[series_col_label],
                f"{raw_col_key}_at_cp":  round(float(mkt[measure_col].iloc[idx]), 2),
                "window_periods":        _CP_WINDOW,
                "series":                window_series,
            })

        cp_probs_series = [
            {
                "date":           str(mkt[date_col].iloc[i]),
                "cp_prob":        float(cp_prob[i]),
                "exp_run_length": float(exp_run_length[i]),
            }
            for i in range(len(cp_prob))
        ]

        result = {
            "ran": True,
            "signals": {
                "n_changepoints":     len(cp_records),
                "cp_candidates":      cp_records,
                "cp_context_windows": cp_context_windows,
                "cp_probs_series":    cp_probs_series,
                "model_params": {
                    "mu_0":             round(mu_0, 6),
                    "beta_0":           round(beta_0, 8),
                    "hazard_lam":       hazard_lam,
                    "date_grain":       grain or "unknown (defaulted to weekly equivalent)",
                    "threshold":        _CP_THRESHOLD,
                    "min_dist":         _CP_MIN_DIST,
                    "target_transform": transform_used,
                },
            },
        }
        if measure_note:
            result["note"] = measure_note
        return result
