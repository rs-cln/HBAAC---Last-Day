"""Check environment, config, raw inputs, and optional LightGBM availability."""

# ruff: noqa: E402

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


CORE_PACKAGES = ["pandas", "numpy", "sklearn", "pyarrow"]
OPTIONAL_PACKAGES = ["lightgbm", "optuna"]


LIBOMP_FIX = """
LightGBM is installed but cannot import because macOS cannot find libomp.dylib.

Option A — Homebrew:
brew install libomp

Option B — Conda:
conda install -c conda-forge libomp lightgbm

Option C — pip/venv:
Install libomp first, then reinstall or retry lightgbm. Make sure libomp is visible to the macOS dynamic linker.
""".strip()


def check_import(package: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(package)
    except Exception as exc:
        return False, str(exc)
    return True, None


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Project root: {ROOT}")

    cfg = config.load_config()
    paths = config.PipelinePaths(cfg=cfg)
    print("Config loaded: config/default.yaml")
    config.print_active_config(cfg)

    missing_core = []
    for package in CORE_PACKAGES:
        ok, error = check_import(package)
        print(f"core package {package}: {'OK' if ok else 'MISSING'}")
        if not ok:
            print(f"  error: {error}")
            missing_core.append(package)

    for package in OPTIONAL_PACKAGES:
        ok, error = check_import(package)
        print(f"optional package {package}: {'OK' if ok else 'UNAVAILABLE'}")
        if package == "lightgbm" and not ok:
            print(f"  error: {error}")
            if error and "libomp.dylib" in error:
                print(LIBOMP_FIX)
        elif not ok:
            print(f"  error: {error}")

    for label, path in [
        ("raw train", paths.train_path),
        ("sample submission", paths.sample_submission_path),
    ]:
        if path.exists():
            print(f"{label}: OK {path} ({path.stat().st_size} bytes)")
        else:
            print(f"{label}: MISSING {path}")
            missing_core.append(str(path))

    if missing_core:
        print(f"Environment check failed. Missing core requirements: {missing_core}")
        return 1
    print("Environment check completed. Core pipeline can run without LightGBM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

