"""InsightsPayload — the LLM-ingestible output artifact."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class InsightsPayload:
    dataset_name: str
    generated_at: str                    # ISO timestamp
    metadata: dict                       # routing, catalog match, table type, star schema
    columns: list[dict]                  # name, dtype, subtype, confidence, profile
    quality_gate: dict                   # overall + per_model decisions + check details
    model_signals: dict                  # {model_name: {"ran": bool, "signals": {...}}}
    transform_context: list[dict]        # column → suggested_transforms
    warnings: list[str]
    knowledge_base_context: dict         # matched datamart + category + description + use_cases

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
