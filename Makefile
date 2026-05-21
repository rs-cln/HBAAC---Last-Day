PYTHON ?= python

.PHONY: check reproduce-champion validate-submission

check:
	$(PYTHON) scripts/00_check_environment.py
	$(PYTHON) -m py_compile src/*.py scripts/*.py

reproduce-champion:
	$(PYTHON) scripts/03_run_full_baseline.py
	$(PYTHON) scripts/02_validate_submission.py submissions/submission.csv

validate-submission:
	$(PYTHON) scripts/02_validate_submission.py submissions/submission.csv
