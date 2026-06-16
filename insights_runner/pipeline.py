"""
insights_runner.pipeline — entry point.

Usage:
    from insights_runner.pipeline import run
    payload = run(connector, router_result, catalog, configs_dir)
    print(payload.to_json())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from named_model_resolution.models import DatamartCatalog, RouterResult
from orchestrator.connectors.base import CatalogConnector

from .output.builder import build
from .output.models import InsightsPayload
from .quality_gate.assessor import QualityAssessor
from .quality_gate.models import QualityReport
from .runners import RUNNER_REGISTRY

_DEFAULT_THRESHOLDS = Path(__file__).parent / "quality_gate" / "thresholds.yaml"
_DEFAULT_SAMPLE_N = 5000


def _aggregate_decision(per_model: dict) -> str:
    decisions = {r.decision for r in per_model.values()}
    if "PASS" in decisions:
        return "PASS"   # at least one model can run
    if "WARN" in decisions:
        return "WARN"
    return "FAIL"


def run(
    connector: CatalogConnector,
    router_result: RouterResult,
    catalog: DatamartCatalog,
    configs_dir: str | Path,
    models_to_run: list[str] | None = None,
    thresholds_path: str | Path | None = None,
    sample_n: int = _DEFAULT_SAMPLE_N,
) -> InsightsPayload:
    """
    Full insights pipeline for a single RouterResult.

    Args:
        connector:       Platform-agnostic data connector (same as router used).
        router_result:   Output from orchestrator.Router.run() for one dataset.
        catalog:         DatamartCatalog (from parse_catalog).
        configs_dir:     Path to pharma_knowledge_base/configs/ (for future use).
        models_to_run:   Restrict to a subset of routed models (None = all).
        thresholds_path: Override path to thresholds.yaml.
        sample_n:        Number of rows to sample from the dataset.

    Returns:
        InsightsPayload (call .to_json() for the LLM-ingestible string).
    """
    configs_dir = Path(configs_dir)
    thresholds_path = Path(thresholds_path) if thresholds_path else _DEFAULT_THRESHOLDS

    # ── Sample data ───────────────────────────────────────────────────────────
    try:
        df: pd.DataFrame = connector.sample_rows(
            router_result.dataset_name, n=sample_n
        )
    except Exception as exc:
        # Return a minimal payload with the error surfaced in warnings
        from datetime import datetime, timezone
        return InsightsPayload(
            dataset_name=router_result.dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
            columns=[],
            quality_gate={"overall_decision": "FAIL", "per_model": {}},
            model_signals={},
            transform_context=[],
            warnings=list(router_result.warnings) + [f"sampling failed: {exc}"],
            knowledge_base_context={},
        )

    assessor = QualityAssessor(thresholds_path)

    model_configs = router_result.model_configs
    if models_to_run:
        model_configs = [mc for mc in model_configs if mc.model_name in models_to_run]

    quality_per_model: dict = {}
    signals_map: dict = {}

    for mc in model_configs:
        # ── Quality gate ──────────────────────────────────────────────────────
        quality = assessor.assess(df, router_result, mc.model_name)
        quality_per_model[mc.model_name] = quality

        if quality.decision == "FAIL":
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": quality.skip_reason or "quality gate FAIL",
            }
            continue

        # ── Run model ─────────────────────────────────────────────────────────
        runner_cls = RUNNER_REGISTRY.get(mc.model_name)
        if runner_cls is None:
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": f"no runner registered for model '{mc.model_name}'",
            }
            continue

        try:
            signals_map[mc.model_name] = runner_cls().run(df, router_result, mc)
        except Exception as exc:
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": f"runner raised an exception: {exc}",
            }

    quality_report = QualityReport(
        overall_decision=_aggregate_decision(quality_per_model) if quality_per_model else "PASS",
        per_model=quality_per_model,
    )

    return build(router_result, quality_report, signals_map, catalog)
