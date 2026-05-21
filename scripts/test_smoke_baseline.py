"""Smoke test baseline CV scoring."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_baseline_pipeline

def main():
    result = run_baseline_pipeline(
        run_mode="smoke",
        score_cv=True,
        clean_smoke_artifacts=True,
    )
    print("Smoke baseline completed.")
    print(f"Submission path: {result['submission_path']}")
    
    # Read CV scores
    cv_scores_path = result['paths'].report_dir / "cv_scores.csv"
    if cv_scores_path.exists():
        import pandas as pd
        df = pd.read_csv(cv_scores_path)
        print("\n=== CV Scores ===")
        print(df.loc[df["slice"] == "official_aggregate"].to_string(index=False))
        
        # Calculate mean CV score for simple_pipeline_sunday_zero
        model_scores = df.loc[(df["slice"] == "official_aggregate") & (df["model_name"] == "simple_pipeline_sunday_zero")]
        print(f"\nMean CV WRMSSE (simple_pipeline_sunday_zero): {model_scores['wrmsse'].mean():.6f}")

if __name__ == "__main__":
    main()
