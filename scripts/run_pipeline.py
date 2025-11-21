import os
from aurora.pipeline.aurora_pipeline import run_aurora_pipeline
from aurora.pipeline.iterative_pipeline import IterativeAuroraPipeline

from aurora.config.settings import (
    DEFAULT_KB_PATH,
    DEFAULT_SCENARIOS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TOP_K_CLAUSES,
    DEFAULT_RISK_THRESHOLD,
)


if __name__ == "__main__":
    kb_path = os.environ.get("AURORA_KB_PATH", DEFAULT_KB_PATH)
    scenarios_path = os.environ.get("AURORA_SCENARIOS_PATH", DEFAULT_SCENARIOS_PATH)
    output_path = os.environ.get("AURORA_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)

    run_aurora_pipeline(
        kb_path=kb_path,
        scenarios_path=scenarios_path,
        output_path=output_path,
        top_k_clauses=DEFAULT_TOP_K_CLAUSES,
        risk_threshold=DEFAULT_RISK_THRESHOLD,
    )
    print(f"AURORA pipeline completed. Audit chains saved to: {output_path}")
