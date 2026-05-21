# Project Context For Next Chat

## Current Champion
- Public LB: 0.50631
- Path: `submissions/champion/public_lb_0_50_champion.csv`
- Hash: `bd159feba71da198a96d319e2f311e47905695eb772a40583c5585bdad8b1139`
- Champion logic: `simple_pipeline_sunday_zero` with weight 1.0; recursive LGBM had weight 0.0.

## Failed Submission
- Submitted file: `submissions/recommended/submit_01_b_global_0_975.csv`
- Public LB: 0.50664, worse than champion.
- Diagnosis: blind global downscale reduced every non-zero non-Sunday prediction and was rejected by Public LB.

## Vietnam Calendar Findings
- Tet demand ratio vs normal weekdays: 0.00147.
- Sunday demand ratio vs normal weekdays: 0.00132.
- Non-Tet public holiday demand ratio vs normal weekdays: 0.233.
- `month_start_end_adjustment` improved top100 profit SKU MAE in 3/3 folds, but h1_28 WRMSSE improved only 1/3 folds.
- Final Sep-Oct 2025 horizon has Sundays, one month-start day, and two month-end days; no Tet or public holidays after 2025-09-06.

## Candidate Created In This Pass
- Name: `candidate_calendar_month_start_end_adjustment`
- Path: `submissions/candidates/candidate_calendar_month_start_end_adjustment.csv`
- Hash: `7e956ff96b56bd1d2f48d26435def0c60a0ccf4ce6315271237293a059f5705a`
- Validation ratio vs champion: 0.9916666647834793
- Evaluation ratio vs champion: 0.9958333323917397
- Top100 profit ratio vs champion: 0.9937499983385559
- Quality gate passed: False
- Submit-ready: False
- Status: audit-only; do not submit without another review.

## Files Modified/Created
- `scripts/08_create_month_start_end_candidate.py`
- `submissions/candidates/candidate_calendar_month_start_end_adjustment.csv`
- `submissions/candidates/candidate_calendar_month_start_end_adjustment.json`
- `reports/month_start_end_candidate_audit.md`
- `reports/month_start_end_candidate_diff.csv`
- `reports/PROJECT_CONTEXT_FOR_NEXT_CHAT.md`
- Validation command also refreshed `reports/submission_checks.md` and `submissions/submission_manifest.validation.json`.

## Protected Files
- `submissions/champion/public_lb_0_50_champion.csv`
- `submissions/champion/public_lb_0_50_champion.json`
- `submissions/submission.csv`
- Raw data under `Data/` and `data/`.

## Next Recommended Action
- Do not submit this candidate yet if the goal is high confidence.
- Review the audit: the validation ratio may be below the submit-consideration gate because two adjusted dates fall in the Public LB window.
- If continuing, compare this candidate against champion on top-profit IDs and consider a narrower top-profit-only month boundary variant in a separate pass.

## Month Start/End Failure Diagnosis
- Existing audit-only candidate: `submissions/candidates/candidate_calendar_month_start_end_adjustment.csv`.
- Candidate hash: `7e956ff96b56bd1d2f48d26435def0c60a0ccf4ce6315271237293a059f5705a`.
- It failed submit-consideration because validation ratio was `0.9916666648`, below the lower gate `0.995`.
- Responsible final-horizon dates/F columns: validation `F25=2025-09-30` month-end, validation `F26=2025-10-01` month-start, evaluation `F28=2025-10-31` month-end.
- Root cause: the 0.90 factor hit 2 of 24 active validation days but only 1 of 24 active evaluation days; Sundays stayed unchanged.
- Best diagnostic-only safer variant from this pass: `half_strength_adjustment`.
- Hypothetical scoring was in-memory only; no new candidate CSV was created.
- Reports: `reports/month_start_end_failure_diagnosis.md`, `reports/month_start_end_hypothetical_scores.csv`.
- Protected files remained unchanged: champion, `submissions/submission.csv`, existing candidate CSV, and raw data.

## Half-Strength Month Start/End Candidate
- Candidate: `submissions/candidates/candidate_calendar_month_start_end_half_strength.csv`.
- Hash: `5fdf855b1aca485269eb453dfe375477530798956ba0734e34c4db9acdcba0b2`.
- Rule: reproduce champion, then apply factor `0.95` only on `2025-09-30`, `2025-10-01`, and `2025-10-31`.
- Ratios vs champion: total `0.996875`, validation `0.9958333333333333`, evaluation `0.9979166666666663`, top100 `0.9968750000000001`.
- Quality gate passed: `True`.
- Status: audit-only, not submit-ready yet.
- Reports: `reports/month_start_end_half_strength_candidate_audit.md`, `reports/month_start_end_half_strength_candidate_diff.csv`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Half-Strength Submit Decision
- Decision: hold; do not submit as the next attempt yet.
- Candidate: `submissions/candidates/candidate_calendar_month_start_end_half_strength.csv`.
- Hash: `5fdf855b1aca485269eb453dfe375477530798956ba0734e34c4db9acdcba0b2`.
- It changes only validation `F25=2025-09-30`, validation `F26=2025-10-01`, and evaluation `F28=2025-10-31` by factor `0.95`.
- Ratios vs champion: total `0.996875`, validation `0.9958333333`, evaluation `0.9979166667`.
- Compared with failed global 0.975, it changes far fewer cells and about one eighth of total absolute movement, but it still reduces two Public LB dates.
- Risk: medium-low overall; hold unless using one submission for a small calendar probe.
- Report: `reports/half_strength_submit_decision.md`.

## Top-Profit Signed Error Overlay Diagnostics
- Diagnostic only; no candidate CSV was created.
- Report: `reports/top_profit_signed_error_diagnostics.md`.
- Scores: `reports/top_profit_overlay_scores.csv`.
- Best overlay idea: `over_minus_5`.
- Quality gate passed: `True`.
- Gate criteria: top100 MAE improves in at least 2/3 folds, h1_28 WRMSSE max worsening <= 0.02, total validation shift <= 1%, touched SKUs <= 200.
- Mean h1_28 WRMSSE delta: `-0.0001631393430402047`.
- Mean top100 MAE delta: `-0.022733632739278125`.
- Public risk: medium-low if turned into one audit candidate.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Top500 Over Minus 5 Candidate
- Candidate: `submissions/candidates/candidate_top500_over_minus_5.csv`.
- Hash: `85ffd99a07533df0614af248d4beaad4eb9dd6329488be38bdd3e58099bf0283`.
- Rule: multiply by `0.95` all forecast cells for top-500-profit SKUs consistently overpredicted in at least 2/3 h1_28 folds.
- Selected SKUs: `100`.
- Ratios vs champion: total `0.9943685915221462`, validation `0.9943685915221462`, evaluation `0.9943685915221462`, top100 `0.9885310398653763`.
- Quality gate passed: `True`.
- Status: audit-only, not submit-ready until separate submit/no-submit decision.
- Reports: `reports/top500_over_minus_5_candidate_audit.md`, `reports/top500_over_minus_5_candidate_diff.csv`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Top500 Over Minus 5 Submit Decision
- Decision: `submit next`.
- Candidate: `submissions/candidates/candidate_top500_over_minus_5.csv`.
- Hash: `85ffd99a07533df0614af248d4beaad4eb9dd6329488be38bdd3e58099bf0283`.
- Reason: targeted -5% correction on consistently overpredicted top-500-profit SKUs; h1_28 WRMSSE and top100 MAE improved 3/3 folds in diagnostics.
- Risk: medium-low Public LB, medium-low Private LB; main red flag is reducing top100 total to `0.9885310398653763` and touching profit-rank 7 SKU-09760.
- Changed nonzero-forecast SKUs: `68`.
- Report: `reports/top500_over_minus_5_submit_decision.md`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Submit Next Upload Prepared
- Recommended upload file: `submissions/recommended/submit_next_top500_over_minus_5.csv`.
- Hash: `85ffd99a07533df0614af248d4beaad4eb9dd6329488be38bdd3e58099bf0283`.
- Source candidate hash matched: `True`.
- Validator passed: shape `(31944, 29)`, id order OK, no duplicates, no NaN/inf, non-negative predictions.
- Upload-ready: `yes` for manual Kaggle upload.
- Report: `reports/submit_next_top500_over_minus_5_upload_check.md`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Public LB Result: Top500 Over Minus 5
- Submitted file: `submissions/recommended/submit_next_top500_over_minus_5.csv`.
- Hash: `85ffd99a07533df0614af248d4beaad4eb9dd6329488be38bdd3e58099bf0283`.
- Public LB: `0.50633`.
- Champion Public LB remains `0.50631`; candidate was worse by `+0.00002`.
- Status: tested, not better than champion.
- Action: stop `top500_over_minus_5` and similar small downscale/overprediction-correction family unless new evidence appears.
- Prior failed broad downscale: global `0.975` scored `0.50664`.
- Month-start/end candidates remain hold/not submit.
- Log updated: `reports/public_lb_submission_log.md`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Top-Profit Underprediction Uplift Diagnostics
- Public lesson: global 0.975 scored `0.50664`; top500 over-minus-5 scored `0.50633`; champion remains `0.50631`. Public LB has not rewarded forecast reductions.
- Tested uplift overlays: under_plus_2/3/5/8 on active top-profit underpredicted SKUs, selected from non-Sunday h1_28 folds.
- Best overlay: `under_plus_8`.
- Gate passed: `False`.
- Candidate created: `False`.
- Candidate path: ``.
- Candidate hash: ``.
- Reports: `reports/top_profit_underprediction_diagnostics.md`, `reports/top_profit_uplift_scores.csv`.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Why Local CV Failed Public LB
- Report: `reports/why_local_cv_failed_public_lb.md`.
- Scope: existing reports only; no new training, no candidates, no submission modifications.
- Global `0.975` failed because it reduced every nonzero non-Sunday champion forecast and Public LB showed the public window did not need uniform downscaling.
- `top500_over_minus_5` failed despite 3/3 local h1_28 and top100 improvements because fold-level SKU residual corrections did not transfer to the final Public LB window; it also reduced top100 demand to `0.988531` of champion and touched profit-rank 7 `SKU-09760`.
- `under_plus_8` was not submitted; it improved h1_28 in only 2/3 folds, worsened top100 MAE in 3/3 folds, and failed gates.
- Single most likely flaw: selection overfitting to unstable fold residuals, especially for top-profit SKUs, under a CV setup that is directionally useful but not calibrated enough for small manual +/- overlays.
- Working conclusion: local CV is directionally useful only. Stop small downscale/overprediction-correction variants unless new evidence appears.

## Existing Artifact Submission Opportunity Audit
- Report: `reports/existing_artifact_submission_opportunity_audit.md`.
- Scope: existing artifacts only; no training, no new candidates, no submission modifications.
- Inspected `submissions/`, `submissions/before_ensemble/`, `submissions/candidates/`, `submissions/recommended/`, `submissions/archive/`, `reports/`, and `models/`.
- Found 40 full valid submission-format CSVs and 4 long-form 56-day forecast CSVs that can be mapped to submission format.
- Champion-equivalent artifacts include `final_ensemble.csv`, `full_baseline_forecast.csv`, `submission_lgbm_full.csv`, `candidate_A_champion_copy.csv`, `candidate_F_dead_365_zero.csv`, and `candidate_F_cold90_quarter.csv`.
- Full LGBM replacement artifact `recursive_lgbm_tweedie_full.csv` is not realistic: validation ratio `1.650924`, evaluation ratio `1.546570`, top100 ratio `1.202524`, and `571392` zero-to-positive cells.
- Best existing non-champion artifact if spending a submission: `submissions/candidates/candidate_D_top100_recent_blend_0_05.csv`.
- Why: structurally different from scalar overlays, small uplift direction after downscale failures, validation/evaluation ratio `1.003172`, top100 ratio `1.009333`, 80 changed SKUs, 3840 changed cells.
- Risk: medium; touches Public window and top-profit SKUs, creates 240 zero-to-positive cells, and still needs a separate submit/no-submit memo before upload.
- Candidate creation: `False`; no new candidate was created.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.

## Submit Next: Top100 Recent Blend 0.05
- Decision report: `reports/candidate_D_top100_recent_blend_submit_decision.md`.
- Upload check: `reports/submit_next_top100_recent_blend_0_05_upload_check.md`.
- Source: `submissions/candidates/candidate_D_top100_recent_blend_0_05.csv`.
- Recommended upload file: `submissions/recommended/submit_next_top100_recent_blend_0_05.csv`.
- Hash: `434de217d95b35b8d2b0abfe7840895816759598a3463fb64307c4ad0f94704b`.
- Decision: `submit next`.
- Risk: `medium`.
- Reason: narrow top100 recent-demand specialist overlay; structurally different from failed downscales; validation/evaluation ratio `1.003172`, top100 ratio `1.009333`, 80 changed SKUs, 3840 changed cells.
- Validation: passed; shape `(31944, 29)`, id order OK, no duplicates, no NaN/inf, all predictions non-negative.
- Sundays changed: `False`.
- Zero-to-positive cells: `240`, from 5 top100-profit SKUs across non-Sunday days. Two have no positive days in last90 but add less than `0.5` units each over 56 days.
- Main risk: touches active top-profit SKUs, especially `SKU-09760` profit rank 7 with `+78.239380` total over 56 days.
- Manual upload only; nothing was submitted automatically.
- Protected files unchanged: champion, `submissions/submission.csv`, and raw data.
