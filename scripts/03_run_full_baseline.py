"""Generate a full-SKU deterministic baseline submission without LightGBM."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import GENERATOR_FULL_BASELINE, run_baseline_pipeline


def main() -> int:
    start = time.perf_counter()
    result = run_baseline_pipeline(
        run_mode="full",
        generator_name=GENERATOR_FULL_BASELINE,
        clean_smoke_artifacts=False,
        score_cv=False,
        build_features_for_smoke=False,
    )
    runtime = time.perf_counter() - start
    summary = result["summary"]
    print("Full baseline generated. This is submit-able but not expected to be competitive.")
    print(f"submission path: {result['submission_path']}")
    print(f"submission shape: {result['submission'].shape}")
    print(f"submission hash: {result['submission_hash']}")
    print(f"prediction min: {summary['min']}")
    print(f"prediction max: {summary['max']}")
    print(f"prediction mean: {summary['mean']}")
    print(f"prediction median: {summary['median']}")
    print(f"validation total prediction: {summary['total_predicted_validation']}")
    print(f"evaluation total prediction: {summary['total_predicted_evaluation']}")
    if result["archived_submission"] is not None:
        print(f"archived previous submission: {result['archived_submission']}")
    print(f"runtime seconds: {runtime:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

