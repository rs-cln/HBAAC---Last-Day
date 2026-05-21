"""Run manifest and artifact tracking utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def utc_timestamp() -> str:
    """Return an ISO UTC timestamp for manifests and archives."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_timestamp() -> str:
    """Return filesystem-safe UTC timestamp."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def hash_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def get_file_metadata(path: Path | str, include_hash: bool = True) -> Dict[str, Any]:
    """Return path, existence, size, modified time, and optional hash."""

    file_path = Path(path)
    payload: Dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
    }
    if not file_path.exists():
        return payload
    stat = file_path.stat()
    payload.update(
        {
            "size_bytes": stat.st_size,
            "modified_time_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds"
            )
            + "Z",
        }
    )
    if include_hash:
        payload["sha256"] = hash_file(file_path)
    return payload


def get_git_commit(root: Path | str = ".") -> Optional[str]:
    """Return current git commit hash when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def get_package_versions() -> Dict[str, Dict[str, Any]]:
    """Return installed package versions and import availability."""

    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
        "pyarrow": "pyarrow",
        "optuna": "optuna",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for import_name, package_name in packages.items():
        item: Dict[str, Any] = {"installed": False, "version": None, "imports": False}
        try:
            item["version"] = metadata.version(package_name)
            item["installed"] = True
        except metadata.PackageNotFoundError:
            out[import_name] = item
            continue
        try:
            __import__(import_name)
            item["imports"] = True
        except Exception as exc:
            item["import_error"] = str(exc)
        out[import_name] = item
    return out


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def artifact_row_count(path: Path | str) -> Optional[int]:
    """Best-effort row count for CSV/parquet artifacts."""

    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        if file_path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(file_path).metadata.num_rows)
            except Exception:
                return int(pd.read_parquet(file_path).shape[0])
        if file_path.suffix == ".csv":
            return int(sum(1 for _ in file_path.open("rb")) - 1)
    except Exception:
        return None
    return None


def collect_artifact_metadata(paths: Iterable[Path | str]) -> Dict[str, Dict[str, Any]]:
    """Collect metadata and row counts for artifacts."""

    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        file_path = Path(path)
        meta = get_file_metadata(file_path, include_hash=False)
        meta["row_count"] = artifact_row_count(file_path)
        out[str(file_path)] = meta
    return out


def submission_stats(submission: pd.DataFrame) -> Dict[str, Any]:
    """Compute prediction stats for F columns."""

    f_cols = [c for c in submission.columns if c.startswith("F")]
    values = submission[f_cols].to_numpy(dtype=float)
    validation = submission.loc[submission["id"].str.endswith("_validation"), f_cols]
    evaluation = submission.loc[submission["id"].str.endswith("_evaluation"), f_cols]
    return {
        "submission_shape": list(submission.shape),
        "min_prediction": float(np.min(values)),
        "max_prediction": float(np.max(values)),
        "mean_prediction": float(np.mean(values)),
        "median_prediction": float(np.median(values)),
        "total_predicted_validation": float(validation.to_numpy(dtype=float).sum()),
        "total_predicted_evaluation": float(evaluation.to_numpy(dtype=float).sum()),
    }


def build_run_manifest(
    *,
    config_values: Mapping[str, Any],
    input_paths: Iterable[Path | str],
    processed_artifact_paths: Iterable[Path | str],
    generator_name: str,
    run_mode: str,
    number_of_skus: Optional[int] = None,
    feature_matrix_shape: Optional[tuple[int, int]] = None,
    submission_path: Optional[Path | str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a reproducibility manifest for a pipeline run."""

    manifest: Dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": get_package_versions(),
        "generator": generator_name,
        "run_mode": run_mode,
        "config": dict(config_values),
        "inputs": {
            str(path): get_file_metadata(path, include_hash=True) for path in input_paths
        },
        "processed_artifacts": collect_artifact_metadata(processed_artifact_paths),
        "number_of_skus_used": number_of_skus,
        "feature_matrix_shape": list(feature_matrix_shape)
        if feature_matrix_shape is not None
        else None,
        "submission_path": str(submission_path) if submission_path is not None else None,
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def build_submission_manifest(
    *,
    submission_path: Path | str,
    submission: pd.DataFrame,
    sample_submission_path: Path | str,
    config_values: Mapping[str, Any],
    generator_name: str,
    forecast_artifact_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build manifest for a generated submission."""

    path = Path(submission_path)
    payload: Dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "generator": generator_name,
        "submission": get_file_metadata(path, include_hash=True),
        "sample_submission": get_file_metadata(sample_submission_path, include_hash=True),
        "config": dict(config_values),
        "forecast_artifact_path": str(forecast_artifact_path)
        if forecast_artifact_path is not None
        else None,
    }
    payload.update(submission_stats(submission))
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
