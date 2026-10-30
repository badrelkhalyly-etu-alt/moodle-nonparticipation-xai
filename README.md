# moodle-nonparticipation-xai

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/badrelkhalyly-etu-alt/moodle-nonparticipation-xai/blob/main/notebooks/reproduce_results.ipynb)

Code for the manuscript *Predicting Course Non-Participation from Moodle Gradebook Traces: A Temporally Valid, Explainable Comparison of Learner-History and Course-Context Signals with an Auditable Decision Framework* (A. Yalid, B. El Khalyly).

## Contents

| Path | Purpose |
|---|---|
| `src/analysis.py` | Complete pipeline: data-quality audit, cohort and features, nested user-grouped cross-validation, user-clustered bootstrap, leave-course-group-out evaluation, sensitivity analyses, out-of-fold SHAP and stability audit, retrospective routing, design-side case study |
| `results/results.json` | Reference output: every number reported in the manuscript |
| `results/figures/` | Figures 1, 3 and 4 |
| `notebooks/reproduce_results.ipynb` | Google Colab notebook: re-runs the pipeline and compares every value with `results/results.json` |
| `data/` | Place the (non-distributed) raw files here, see `data/README.md` |

## Reproduce

```bash
git clone https://github.com/badrelkhalyly-etu-alt/moodle-nonparticipation-xai.git
cd moodle-nonparticipation-xai
pip install -r requirements.txt
python src/analysis.py --data-dir data --out results/recomputed.json
```

or open `notebooks/reproduce_results.ipynb` in Google Colab and run all cells. A full run takes about 15 minutes on a standard CPU. Use `--fast` for a quick smoke test (reduced random-forest grid, 200 bootstrap resamples; values will differ).

Library versions are pinned in `requirements.txt`. Tree ensembles and cross-validation splits can change slightly across library versions, so use the pinned versions to obtain identical numbers.

## Design summary

- Cutoff `T0` = 2022-11-24 00:00 UTC, immediately before the administrative batch that wrote 89.9% of the non-attempt markers; sensitivity cutoff `T1` = 2022-12-01.
- Outcome: course total = -1 (no graded participation), per user-course pair whose course total was written at or after the cutoff.
- 10 learner-history features and 4 leave-self-out course-context features, all from records strictly before the cutoff.
- Outer 5-fold `StratifiedGroupKFold` grouped by user (seed 42); inner grouped 3-fold grid search by Macro-F1; median imputation and scaling fitted in training folds; balanced class weights.
- Leave-course-group-out: courses partitioned deterministically into 5 groups of near-equal size (greedy assignment, independent of library version).
# moodle-nonparticipation-xai
