"""Utility helpers for reproducible pipeline runs."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from . import config


def get_logger(name: str = "hbaac") -> logging.Logger:
    """Return a configured logger without duplicating handlers."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directories(paths: Optional[config.PipelinePaths] = None) -> None:
    """Create the project directories expected by the pipeline."""

    paths = paths or config.PipelinePaths()
    for directory in [
        paths.raw_dir,
        paths.processed_dir,
        paths.feature_dir,
        paths.model_dir / "recursive",
        paths.model_dir / "direct",
        paths.model_dir / "hurdle",
        paths.model_dir / "final",
        paths.report_dir,
        paths.submission_dir,
        paths.archive_dir,
        paths.before_ensemble_dir,
        paths.notebook_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def run_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_metadata(
    artifact_path: Path,
    run_mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a sidecar metadata file for an artifact."""

    metadata = {
        "artifact": str(artifact_path),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_mode": run_mode,
    }
    if extra:
        metadata.update(extra)
    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")
    write_json(meta_path, metadata)
    return meta_path


def copy_legacy_raw_files(paths: Optional[config.PipelinePaths] = None) -> None:
    """Copy raw files from legacy `Data/` into `data/raw/` when needed."""

    paths = paths or config.PipelinePaths()
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in [config.TRAIN_FILE, config.SAMPLE_SUBMISSION_FILE]:
        dst = paths.raw_dir / filename
        src = paths.legacy_raw_dir / filename
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)


def require_columns(columns: Iterable[str], required: Iterable[str], context: str) -> None:
    """Raise a clear error if a table is missing required columns."""

    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def finite_nonnegative_frame(df: Any, value_columns: Iterable[str], context: str) -> None:
    """Assert that prediction columns are finite and non-negative."""

    values = df[list(value_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"{context} contains non-finite prediction values.")
    if (values < 0).any():
        raise AssertionError(f"{context} contains negative prediction values.")
