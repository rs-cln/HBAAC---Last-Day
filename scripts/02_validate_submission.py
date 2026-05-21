"""Validate a Kaggle submission against sample_submission.csv."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data import load_sample_submission
from src.inference import make_submission_from_56day_forecast, validate_submission
from src.run_manifest import build_submission_manifest, hash_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_path",
        nargs="?",
        default="submissions/submission.csv",
        help="Submission CSV path to validate.",
    )
    return parser.parse_args()


def validate_horizon_mapping_if_possible(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    manifest_path: Path,
) -> str:
    if not manifest_path.exists():
        return "No submission manifest found; horizon artifact mapping check skipped."
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_submission = manifest.get("submission", {}).get("path")
    if manifest_submission:
        manifest_submission_path = Path(manifest_submission).resolve()
        # Candidate files are intentionally compared against the sample format, not
        # against the current champion forecast artifact.
        if manifest_submission_path != Path(submission.attrs.get("source_path", "")).resolve():
            return "Submission path differs from manifest submission; horizon artifact mapping check skipped."
    forecast_path_value = manifest.get("forecast_artifact_path")
    if not forecast_path_value:
        return "No forecast artifact path in submission manifest; horizon mapping check skipped."
    forecast_path = Path(forecast_path_value)
    if not forecast_path.exists():
        return f"Forecast artifact missing; horizon mapping check skipped: {forecast_path}"
    forecast = pd.read_csv(forecast_path, parse_dates=["Date"])
    expected = make_submission_from_56day_forecast(forecast, sample, output_path=None)
    f_cols = [f"F{i}" for i in range(1, 29)]
    if not np.allclose(
        submission[f_cols].to_numpy(dtype=float),
        expected[f_cols].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise AssertionError(
            "Submission values do not match h1-h28/h29-h56 mapping from forecast artifact."
        )
    return f"Horizon mapping matches forecast artifact: {forecast_path}"


def write_checks_report(
    path: Path,
    stats: dict,
    top_ids: pd.DataFrame,
    horizon_message: str,
    validation_evidence: dict,
) -> None:
    lines = [
        "# Submission Checks",
        "",
        "## Validation Evidence",
    ]
    for key, value in validation_evidence.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Prediction Summary",
            f"- min: {stats['min']}",
            f"- max: {stats['max']}",
            f"- mean: {stats['mean']}",
            f"- median: {stats['median']}",
            f"- total validation prediction: {stats['total_validation']}",
            f"- total evaluation prediction: {stats['total_evaluation']}",
            "",
            "## Horizon Mapping",
            f"- {horizon_message}",
            "",
            "## Top 20 IDs By Predicted Demand",
        ]
    )
    for row in top_ids.to_dict(orient="records"):
        lines.append(f"- {row['id']}: {row['predicted_total']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = config.load_config()
    paths = config.PipelinePaths(cfg=cfg)
    submission_path = Path(args.submission_path)
    if not submission_path.is_absolute():
        submission_path = ROOT / submission_path
    sample = load_sample_submission(paths.sample_submission_path)
    submission = pd.read_csv(submission_path)
    submission.attrs["source_path"] = str(submission_path)
    validate_submission(submission, sample)

    f_cols = [f"F{i}" for i in range(1, 29)]
    values = submission[f_cols].to_numpy(dtype=float)
    validation_rows = submission["id"].str.endswith("_validation")
    evaluation_rows = submission["id"].str.endswith("_evaluation")
    validation_evidence = {
        "shape": tuple(submission.shape),
        "id_order_matches_sample_submission": submission["id"].equals(sample["id"]),
        "no_duplicate_id": bool(submission["id"].is_unique),
        "non_negative_predictions": bool((values >= 0).all()),
        "no_nan": bool(not np.isnan(values).any()),
        "no_inf": bool(np.isfinite(values).all()),
        "validation_rows": int(validation_rows.sum()),
        "evaluation_rows": int(evaluation_rows.sum()),
    }
    horizon_message = validate_horizon_mapping_if_possible(
        submission,
        sample,
        paths.submission_dir / "submission_manifest.json",
    )
    stats = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "total_validation": float(submission.loc[validation_rows, f_cols].sum().sum()),
        "total_evaluation": float(submission.loc[evaluation_rows, f_cols].sum().sum()),
    }
    top_ids = submission.assign(predicted_total=submission[f_cols].sum(axis=1))[
        ["id", "predicted_total"]
    ].sort_values("predicted_total", ascending=False).head(20)
    write_checks_report(
        paths.report_dir / "submission_checks.md",
        stats,
        top_ids,
        horizon_message,
        validation_evidence,
    )
    manifest = build_submission_manifest(
        submission_path=submission_path,
        submission=submission,
        sample_submission_path=paths.sample_submission_path,
        config_values=cfg,
        generator_name="scripts/02_validate_submission.py",
    )
    manifest["submission"]["sha256"] = hash_file(submission_path)
    write_json(paths.submission_dir / "submission_manifest.validation.json", manifest)

    print(f"submission path: {submission_path}")
    print(f"submission hash: {hash_file(submission_path)}")
    print(f"shape: {tuple(submission.shape)}")
    print(f"id order matches sample_submission.csv: {validation_evidence['id_order_matches_sample_submission']}")
    print(f"no duplicate id: {validation_evidence['no_duplicate_id']}")
    print(f"non-negative predictions: {validation_evidence['non_negative_predictions']}")
    print(f"no NaN: {validation_evidence['no_nan']}")
    print(f"no inf: {validation_evidence['no_inf']}")
    print(f"min: {stats['min']}")
    print(f"max: {stats['max']}")
    print(f"mean: {stats['mean']}")
    print(f"median: {stats['median']}")
    print(f"total validation prediction: {stats['total_validation']}")
    print(f"total evaluation prediction: {stats['total_evaluation']}")
    print(horizon_message)
    print("top 20 IDs by total predicted demand:")
    print(top_ids.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
