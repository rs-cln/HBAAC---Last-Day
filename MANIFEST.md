# Release Manifest

## Included
- `src/*.py`: reusable pipeline, preprocessing, baseline, metric, inference, and manifest helpers.
- `scripts/00_check_environment.py`: lightweight environment and input check.
- `scripts/03_run_full_baseline.py`: deterministic full-SKU champion baseline runner.
- `scripts/02_validate_submission.py`: submission validator.
- `config/default.yaml`: sanitized config using `Data/raw/` and disabling LightGBM.
- `requirements.txt`: dependency list from the working project, if present.
- `Makefile`: release-specific convenience commands.
- `reports/PROJECT_CONTEXT_FOR_NEXT_CHAT.md`: project handoff context.
- `README.md`, `REPRODUCE_CHAMPION.md`, `MANIFEST.md`, `.gitignore`.

## Excluded
- Raw Kaggle data.
- Processed parquet/csv feature artifacts.
- Model files.
- `.venv`.
- Existing `submissions/submission.csv`.
- Candidate and champion CSV files.

## Champion Metadata
- Public LB: `0.50566`
- Champion hash: `e6ced5b9990d843c9e38eea70fe95afe5b2fd9f9e61d0d1676592727da92bb06`
- Method: `simple_pipeline_sunday_zero` (optimized with 28-day blend for top 100 SKUs & month boundary downscaling for Class A)

## Reproduce Command

```bash
python scripts/00_check_environment.py
python scripts/03_run_full_baseline.py
python scripts/02_validate_submission.py submissions/submission.csv
```
