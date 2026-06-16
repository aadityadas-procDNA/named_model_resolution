"""Quality gate data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityCheckResult:
    check_name: str
    status: str           # "PASS" | "WARN" | "FAIL"
    detail: str           # human-readable one-liner for LLM context
    metric: float | None  # numeric value (e.g. fill_rate=0.94, max_r=0.87)


@dataclass
class ModelQualityReport:
    model_name: str
    decision: str                          # "PASS" | "WARN" | "FAIL"
    checks: list[QualityCheckResult] = field(default_factory=list)
    skip_reason: str | None = None         # populated on FAIL


@dataclass
class QualityReport:
    overall_decision: str                           # "PASS" | "WARN" | "FAIL"
    per_model: dict[str, ModelQualityReport] = field(default_factory=dict)
