# Reproduce Champion Baseline

## Prerequisites
Install dependencies in your own environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy Kaggle data into:

```text
Data/raw/train.csv
Data/raw/sample_submission.csv
```

## Command
Run:

```bash
python scripts/00_check_environment.py
python scripts/03_run_full_baseline.py
python scripts/02_validate_submission.py submissions/submission.csv
```

The baseline script writes:

```text
submissions/submission.csv
reports/run_manifest.json
reports/submission_checks.md
submissions/submission_manifest.json
```

## Expected Method
- `run_mode`: `full`
- `target_variant`: `net_daily_qty_clip0`
- `baseline_method`: `simple_pipeline_sunday_zero`
- `forecast_horizon`: `56`
- `public_horizon`: `28`

The champion hash from the original private workspace was:

```text
bd159feba71da198a96d319e2f311e47905695eb772a40583c5585bdad8b1139
```

Hash equality can depend on library versions and float formatting. Always run the validator before upload.
