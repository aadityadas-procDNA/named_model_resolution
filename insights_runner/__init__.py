"""
insights_runner — Quality gate + model bridge layer between named_model_resolution
and the insights_generation pipeline.

Entry point:
    from insights_runner.pipeline import run
    payload = run(connector, router_result, catalog, configs_dir)
    print(payload.to_json())
"""
