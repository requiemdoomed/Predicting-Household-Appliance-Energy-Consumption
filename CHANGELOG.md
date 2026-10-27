# Version 2.0.0 - October 27, 2026

## Reproduced defects, then fixes

Three targeted regression tests were run against the original implementation and all failed. After the fixes, the same three tests passed:

- An empty CSV with valid headers was accepted by the loader. It now raises a clear ValueError before training.
- The source checksum was always read from the default CSV, even when another dataset was loaded. The loader now attaches the actual source-file checksum; the experiment also records a fingerprint of the analyzed DataFrame.
- Rolling forecasting used numeric index labels to decide training membership. Changing otherwise identical row labels emptied the training set. It now normalizes row labels before making positional splits.

Additional hardening rejects malformed inference inputs, validates stored recipe parameters against the declared search, creates missing forecast output directories and prevents the CLI from overwriting its input file.

## Experiments

- Thirteen environmental and seventeen history-forecast configurations.
- Four-fold final temporal search, plus three outer folds with three-fold inner selection and one-day gaps.
- Every selected model is fitted on real UCI observations.
- Shifted history-window features, regularized boosting and extra trees.
- Exploratory paired bootstrap sensitivity at 6, 24 and 48 hours.
- Full candidate results, nested predictions, daily errors, artifact hashes and protocol record.
- Actual pytest log and JUnit XML; full notebook re-execution.

## Interpretation

Forecasting improved; environmental expansion did not improve the inspected final period. The original ridge stays the default condition estimator. Experimental environmental extra trees is opt-in. The new history model is selected by development CV, not by picking the best final-period score.

The original final period was already inspected in version 1. Version 2 comparisons on that period are retrospective. New external measurements are needed for a genuinely independent final evaluation. No perfect-performance claim is made.
