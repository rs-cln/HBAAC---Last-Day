# HBAAC Champion Release

This is a GitHub-safe release folder for reproducing the current HBAAC champion-style deterministic baseline.

## Champion
- Champion Public LB: `0.50631`
- Champion hash: `bd159feba71da198a96d319e2f311e47905695eb772a40583c5585bdad8b1139`
- Champion logic: `simple_pipeline_sunday_zero`
- Raw Kaggle data is **not included**.

## Data Required
Place the Kaggle files here:

```text
Data/raw/train.csv
Data/raw/sample_submission.csv
```

Do not upload private, raw, processed, or generated competition data to GitHub.

## Reproduce
From this folder:

```bash
python scripts/00_check_environment.py
python scripts/03_run_full_baseline.py
python scripts/02_validate_submission.py submissions/submission.csv
```

The generated `submissions/submission.csv` should be validated before any Kaggle upload.

## Notes
- This release intentionally excludes raw data, processed data, models, `.venv`, and existing `submission.csv`.
- Generated outputs are ignored by `.gitignore`.
- LightGBM is not needed for the champion simple baseline reproduction.
