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
- Public LB: `0.50631`
- Champion hash: `bd159feba71da198a96d319e2f311e47905695eb772a40583c5585bdad8b1139`
- Method: `simple_pipeline_sunday_zero`

## Reproduce Command

```bash
python scripts/00_check_environment.py
python scripts/03_run_full_baseline.py
python scripts/02_validate_submission.py submissions/submission.csv
```
