"""Reproducible baseline pipeline orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import config
from .baselines import generate_all_baselines, score_baselines_for_folds
from .data import (
    compute_profit_weights,
    extract_submission_skus,
    load_sample_submission,
    load_transactions,
    save_preprocessing_artifacts,
    select_smoke_skus,
)
from .demand_classification import build_demand_classification
from .features import build_feature_frame, model_feature_columns
from .inference import (
    make_submission_from_56day_forecast,
    prediction_summary,
    write_submission_checks,
)
from .run_manifest import (
    build_run_manifest,
    build_submission_manifest,
    hash_file,
    safe_timestamp,
    write_json,
)
from .utils import copy_legacy_raw_files, ensure_directories, set_seed, write_metadata


GENERATOR_SMOKE = "scripts/01_run_smoke_baseline.py::run_baseline_pipeline"
GENERATOR_FULL_BASELINE = "scripts/03_run_full_baseline.py::run_baseline_pipeline"


def archive_submission_if_needed(paths: config.PipelinePaths, cfg: Dict[str, Any]) -> Optional[Path]:
    """Archive existing output submission before replacing it."""

    output_path = paths.output_submission_path
    champion_dir = paths.submission_dir / "champion"
    try:
        output_path.relative_to(champion_dir)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"Refusing to write a protected champion path: {output_path}")
    should_archive = bool(cfg["submission"].get("archive_existing_submission", True))
    if not output_path.exists() or not should_archive:
        return None
    paths.archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = paths.archive_dir / f"submission_{safe_timestamp()}.csv"
    shutil.copy2(output_path, archive_path)
    return archive_path


def safe_remove_smoke_artifacts(paths: config.PipelinePaths, target_variant: str) -> None:
    """Remove only artifacts that are clearly marked as smoke-generated."""

    candidates = [
        paths.processed_dir / "daily_sparse.parquet",
        paths.processed_dir / "daily_panel_active_span.parquet",
        paths.processed_dir / "daily_panel_full_span.parquet",
        paths.processed_dir / "sku_profile.parquet",
        paths.processed_dir / "profit_weights.parquet",
        paths.processed_dir / "demand_classification.parquet",
        paths.processed_dir / "baseline_fold_predictions.parquet",
        paths.feature_dir / f"features_{target_variant}.parquet",
        paths.report_dir / "cv_scores.csv",
        paths.report_dir / "ablation_results.csv",
        paths.report_dir / "ensemble_weights.csv",
        paths.before_ensemble_dir / "simple_pipeline_sunday_zero.csv",
    ]
    for artifact in candidates:
        _remove_if_smoke_artifact(artifact)


def _remove_if_smoke_artifact(path: Path) -> None:
    if not path.exists():
        return
    metadata_path = Path(str(path) + ".metadata.json")
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if meta.get("run_mode") == "smoke":
            path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        return
    if "smoke" in path.name:
        path.unlink(missing_ok=True)


def assert_no_protected_artifacts_for_smoke(
    paths: config.PipelinePaths,
    target_variant: str,
) -> None:
    """Refuse smoke runs that would overwrite artifacts marked as non-smoke."""

    candidates = [
        paths.processed_dir / "daily_sparse.parquet",
        paths.processed_dir / "daily_panel_active_span.parquet",
        paths.processed_dir / "daily_panel_full_span.parquet",
        paths.processed_dir / "sku_profile.parquet",
        paths.processed_dir / "profit_weights.parquet",
        paths.processed_dir / "demand_classification.parquet",
        paths.processed_dir / "baseline_fold_predictions.parquet",
        paths.feature_dir / f"features_{target_variant}.parquet",
        paths.report_dir / "cv_scores.csv",
        paths.before_ensemble_dir / "simple_pipeline_sunday_zero.csv",
    ]
    protected = []
    for artifact in candidates:
        metadata_path = Path(str(artifact) + ".metadata.json")
        if not artifact.exists() or not metadata_path.exists():
            continue
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            protected.append(str(artifact))
            continue
        if meta.get("run_mode") not in {None, "smoke"}:
            protected.append(str(artifact))
    if protected:
        raise RuntimeError(
            "Smoke run would overwrite non-smoke artifacts. Refusing to continue: "
            + ", ".join(protected)
        )


def run_baseline_pipeline(
    *,
    run_mode: str,
    config_path: Optional[Path | str] = None,
    generator_name: Optional[str] = None,
    clean_smoke_artifacts: bool = False,
    score_cv: bool = True,
    build_features_for_smoke: bool = True,
) -> Dict[str, Any]:
    """Run deterministic baseline pipeline and write manifests."""

    cfg = config.load_config(config_path, run_mode=run_mode)
    paths = config.PipelinePaths(cfg=cfg)
    ensure_directories(paths)
    copy_legacy_raw_files(paths)
    set_seed(int(cfg["run_control"]["random_seed"]))
    target_variant = str(cfg["run_control"]["target_variant"])
    baseline_method = str(cfg["run_control"]["baseline_method"])
    mode_cfg = config.mode_config(run_mode, cfg)

    if run_mode == "smoke" and clean_smoke_artifacts:
        safe_remove_smoke_artifacts(paths, target_variant)
        assert_no_protected_artifacts_for_smoke(paths, target_variant)

    archived_submission = archive_submission_if_needed(paths, cfg)

    sample = load_sample_submission(paths.sample_submission_path)
    transactions, clean_report = load_transactions(paths.train_path)
    sample_skus = extract_submission_skus(sample)
    weights_full = compute_profit_weights(transactions, sample_skus)

    sku_subset = None
    if run_mode == "smoke":
        sku_subset = select_smoke_skus(
            list(sample_skus),
            weights_full,
            n_skus=int(mode_cfg["n_skus"]),
            top_profit_skus=int(mode_cfg["top_profit_skus"]),
        )

    preprocessing_outputs = save_preprocessing_artifacts(
        transactions=transactions,
        sample=sample,
        run_mode=run_mode,
        sku_subset=sku_subset,
        paths=paths,
    )
    panel_active = pd.read_parquet(paths.processed_dir / "daily_panel_active_span.parquet")
    panel_full = pd.read_parquet(paths.processed_dir / "daily_panel_full_span.parquet")
    weights = pd.read_parquet(paths.processed_dir / "profit_weights.parquet")

    feature_shape = None
    demand_path = paths.processed_dir / "demand_classification.parquet"
    demand = build_demand_classification(panel_active, target_col=target_variant, weight_table=weights)
    demand.to_parquet(demand_path, index=False)
    write_metadata(demand_path, run_mode, {"target_col": target_variant, "rows": len(demand)})

    should_build_features = build_features_for_smoke if run_mode == "smoke" else False
    feature_path = paths.feature_dir / f"features_{target_variant}.parquet"
    if should_build_features:
        features = build_feature_frame(
            panel_active,
            target_col=target_variant,
            demand_profile=demand,
            weight_table=weights,
        )
        feature_cols = model_feature_columns(features, target_variant)
        features.to_parquet(feature_path, index=False)
        feature_shape = tuple(features.shape)
        write_metadata(
            feature_path,
            run_mode,
            {
                "target_col": target_variant,
                "rows": len(features),
                "feature_count": len(feature_cols),
            },
        )

    cv_scores_path = paths.report_dir / "cv_scores.csv"
    baseline_predictions_path = paths.processed_dir / "baseline_fold_predictions.parquet"
    if score_cv:
        scores, fold_predictions = score_baselines_for_folds(
            panel_full,
            weights,
            target_col=target_variant,
        )
        scores.to_csv(cv_scores_path, index=False)
        fold_predictions.to_parquet(baseline_predictions_path, index=False)
        write_metadata(
            cv_scores_path,
            run_mode,
            {"target_col": target_variant, "rows": len(scores)},
        )
        write_metadata(
            baseline_predictions_path,
            run_mode,
            {"target_col": target_variant, "rows": len(fold_predictions)},
        )
        ablation = (
            scores.loc[scores["slice"].eq("h29_56")]
            .groupby("model_name", as_index=False)["wrmsse"]
            .mean()
            .sort_values("wrmsse")
        )
        ablation.to_csv(paths.report_dir / "ablation_results.csv", index=False)
    else:
        ablation = pd.DataFrame([{"model_name": baseline_method, "wrmsse": None}])
        ablation.to_csv(paths.report_dir / "ablation_results.csv", index=False)

    pd.DataFrame([{"model_name": baseline_method, "weight": 1.0}]).to_csv(
        paths.report_dir / "ensemble_weights.csv",
        index=False,
    )

    forecast_skus = sorted(panel_full["ItemCode"].unique())
    history = panel_full.loc[panel_full["Date"] <= pd.Timestamp(cfg["competition"]["train_end"])]
    baselines = generate_all_baselines(
        history,
        forecast_skus,
        pd.Timestamp(cfg["competition"]["validation_start"]),
        target_variant,
        horizon=int(cfg["competition"]["forecast_horizon"]),
    )
    if baseline_method not in baselines:
        raise KeyError(
            f"Configured baseline_method={baseline_method!r} not in {sorted(baselines)}"
        )
    forecast = baselines[baseline_method].drop(columns=["model_name"])
    forecast_path = paths.before_ensemble_dir / f"{baseline_method}.csv"
    forecast.to_csv(forecast_path, index=False)
    write_metadata(
        forecast_path,
        run_mode,
        {"baseline_method": baseline_method, "rows": len(forecast)},
    )

    submission = make_submission_from_56day_forecast(
        forecast,
        sample,
        output_path=paths.output_submission_path,
        run_mode=run_mode,
    )
    summary = prediction_summary(
        submission,
        sample,
        history_panel=panel_full,
        target_col=target_variant,
        weight_table=weights,
    )
    write_submission_checks(paths.report_dir / "submission_checks.md", summary)

    processed_artifacts = [
        *preprocessing_outputs.values(),
        demand_path,
        forecast_path,
        paths.report_dir / "ablation_results.csv",
        paths.report_dir / "ensemble_weights.csv",
    ]
    if feature_path.exists():
        processed_artifacts.append(feature_path)
    if score_cv:
        processed_artifacts.extend([cv_scores_path, baseline_predictions_path])

    generator = generator_name or (
        GENERATOR_SMOKE if run_mode == "smoke" else GENERATOR_FULL_BASELINE
    )
    run_manifest = build_run_manifest(
        config_values=cfg,
        input_paths=[paths.train_path, paths.sample_submission_path],
        processed_artifact_paths=processed_artifacts,
        generator_name=generator,
        run_mode=run_mode,
        number_of_skus=int(panel_full["ItemCode"].nunique()),
        feature_matrix_shape=feature_shape,
        submission_path=paths.output_submission_path,
        extra={
            "baseline_method": baseline_method,
            "target_variant": target_variant,
            "clean_report": clean_report,
            "archived_submission": str(archived_submission)
            if archived_submission is not None
            else None,
        },
    )
    write_json(paths.report_dir / "run_manifest.json", run_manifest)

    submission_manifest = build_submission_manifest(
        submission_path=paths.output_submission_path,
        submission=submission,
        sample_submission_path=paths.sample_submission_path,
        config_values=cfg,
        generator_name=generator,
        forecast_artifact_path=forecast_path,
    )
    write_json(paths.submission_dir / "submission_manifest.json", submission_manifest)

    return {
        "config": cfg,
        "paths": paths,
        "submission": submission,
        "summary": summary,
        "submission_hash": hash_file(paths.output_submission_path),
        "submission_path": paths.output_submission_path,
        "forecast_path": forecast_path,
        "archived_submission": archived_submission,
        "run_manifest": run_manifest,
        "submission_manifest": submission_manifest,
        "ablation": ablation,
    }
