# WattWise

### Household appliance energy prediction, with an honest test of temporal generalization

**Version 2.0.0 · Author: Swayam Shakya** · Python · Jupyter · Reproducible regression research

![Observed energy through time](results/figures/timeline.png)

WattWise investigates a focused question: **How well can multiple linear regression estimate appliance energy consumption from environmental measurements on a later period?** It uses the real UCI Appliances Energy Prediction dataset and keeps the full analysis inspectable, from raw measurements to row-level predictions.

The original study adds calendar features, ridge regularization, a nonlinear benchmark, temporal cross-validation, residual diagnostics, uncertainty analysis, and a separately defined rolling forecast. It is a research project with measured strengths and limitations, not a production energy-management product.

## Start here

1. Extract the complete repository folder. Keep its directories together.
2. Install **Python 3.12** and open a terminal in this folder.
3. Create and activate an environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab WattWise.ipynb
```

Choose **Kernel → Restart Kernel and Run All Cells**. The raw dataset is included; internet is only needed for dependency installation or optional dataset restoration. Open the notebook from the repository directory, not as a standalone downloaded file. The project has no API keys, account requirement or paid service.

If your notebook uses another environment, register this one and select it:

```bash
python -m ipykernel install --user --name wattwise --display-name "Python (WattWise)"
```

## Version 2 results: improvement without hiding regressions

The audit fixed three reproduced defects and evaluated **30 model configurations** with nested chronological validation. All tuning uses development data. Because the final period was already inspected in version 1, new results on it are **retrospective comparisons**, not a newly untouched test.

| Task and recipe | MAE (Wh) | RMSE (Wh) | R² |
| --- | ---: | ---: | ---: |
| Environmental MLR, original | 46.933 | 85.686 | 0.114 |
| Environmental ridge, original/default | 47.222 | 84.708 | 0.134 |
| Environmental extra trees, v2 CV winner | 61.365 | 91.187 | -0.003 |
| Forecast persistence | 26.651 | 66.438 | 0.467 |
| Forecast lagged ridge, original | 26.897 | 60.994 | 0.551 |
| Forecast history ridge, v2 | 27.659 | 59.695 | 0.570 |
| Forecast history boosting, v2 | 25.307 | 58.001 | 0.594 |
| Forecast history extra trees, v2 CV winner | 25.506 | 57.230 | 0.605 |

The new forecasting model reduces RMSE by **6.17%** relative to the original lagged ridge and **13.86%** relative to persistence. Boosting has slightly lower MAE; extra trees wins the declared development RMSE criterion. These units are **Wh per ten-minute interval**, not watts, and R² is not classification accuracy.

The environmental expansion **did not improve** the retrospective final-period score. The development-selected tree model is retained as an explicitly experimental option, and the original ridge remains the default environmental predictor. This does not change which model won the new CV search.

![Version 2 comparisons](results/enhanced/comparison.png)

Three outer chronological folds rerun the entire family-and-parameter selection process using three inner folds. The selected history extra-tree model improves over persistence on every outer validation window:

| Outer fold | Selected-model RMSE (Wh) | Persistence RMSE (Wh) |
| --- | ---: | ---: |
| 1 | 66.618 | 72.361 |
| 2 | 61.331 | 70.863 |
| 3 | 60.925 | 72.279 |

These are internal development-validation results, not external household validation. Full candidate and fold evidence is in `results/enhanced/`. The original experiment is preserved in the notebook and `results/v1_reference/`.

**Keep the two tasks separate.** Environmental regression uses recorded same-interval conditions, including interpolated weather. Rolling forecasting uses only previous energy readings and known calendar values. It consumes later measurements only as they become observable; it is not a month-ahead prediction from one origin.

## Research safeguards

- An 80/20 chronological split; no shuffled final evaluation.
- Four expanding development folds with a one-day gap for final tuning, plus three outer folds that repeat the full selection procedure.
- Scaling inside each fold; test data do not set means, scales or parameters.
- Random controls and lighting energy excluded from the main features.
- Mean and median baselines, plus persistence for the forecasting extension.
- Raw negative predictions reported, not silently clipped before evaluation.
- Training-defined peak threshold, standardized coefficients and VIF diagnostics.
- Paired block bootstrap: original MLR comparison plus v2 sensitivity at 6-, 24- and 48-hour blocks. These intervals are conditional and exploratory.
- Recorded dataset hash, package versions, fold boundaries and complete predictions.

The design is explicit in the notebook, paper and technical report. The bootstrap interval is conditional on these fitted models and this window; it is not a universal guarantee. The tuned CV score is a selection score, not a nested-CV estimate.

## Commands

```bash
# Run training and evaluation without a notebook
python run_experiments.py

# Train version 2 and run nested validation (allow several minutes)
python run_enhancements.py

# Run the automated tests and keep machine-readable evidence
python -m pytest -q --junitxml=results/test_results.xml

# Execute and save all notebook outputs in a normal Jupyter kernel
python scripts/execute_notebook.py

# Alternative for hosts that prohibit kernel sockets
python scripts/execute_notebook.py --in-process

# Estimate same-interval energy for the supplied example measurements
python predict.py data/example_conditions.csv results/new_predictions.csv

# Forecast the next interval from actual energy history
python forecast.py data/example_history.csv results/next_interval.csv

# Explicitly opt into the experimental environment model
python predict.py data/example_conditions.csv results/experimental.csv --experimental-v2

# Rebuild both documents from the recorded experiment files
python scripts/build_reports.py

# Restore the exact original dataset if it was removed
python scripts/download_data.py
```

By default, the prediction command fits the fixed original ridge recipe on development data and writes raw and nonnegative estimates, plus a simple flag for predictors outside their training range. It validates missing and non-finite inputs. This flag is not a calibrated uncertainty estimate. The sample inputs are three real held-out rows with the target removed, not an external validation dataset.

The forecasting CSV needs `date` and `Appliances` columns with at least 144 uninterrupted ten-minute readings. Use ISO 8601 timestamps. The next timestamp is derived automatically; the current target is unknown. The supplied history is a real held-out sequence. Both CLI tools reject overwriting their own input files. No pickle model is loaded: selected configurations are validated against the declared search before fitting.

## Repository map

| Path | Purpose |
| --- | --- |
| `WattWise.ipynb` | Main explanatory notebook with real captured outputs |
| `src/energy.py` | Validation, features, fitting, metrics, diagnostics and figures |
| `run_experiments.py` | Original reference experiment runner |
| `run_enhancements.py`, `src/enhanced.py` | Version 2 searches, history features and nested evaluation |
| `forecast.py`, `src/inference.py` | Validated next-interval and selected-recipe inference |
| `predict.py` | Example CSV prediction interface |
| `data/raw/energydata_complete.csv` | Original attributed public dataset |
| `data/provenance.json` | Source URL, license, retrieval date and SHA-256 |
| `data/example_conditions.csv` | Real sensor rows for the example command |
| `results/metrics.csv` | Primary holdout metrics and tuning scores |
| `results/predictions.csv` | Primary predictions for every held-out row |
| `results/cv_folds.csv` | Chosen configurations' validation scores and dates |
| `results/forecast_*` | Separate rolling forecasting results and configuration |
| `results/coefficients.csv`, `results/vif.csv` | Interpretability diagnostics |
| `results/diagnostics.json`, `results/manifest.json` | Residual analysis, bootstrap, versions and chronology |
| `results/figures/` | Six reproducible research figures |
| `reports/research_paper.pdf` | Research manuscript with version 2 results |
| `reports/technical_report.pdf` | Detailed implementation report with audit and new experiments |
| `reports/*.md` | Editable document sources |
| `scripts/build_reports.py` | Report regeneration from experiment outputs |
| `tests/` | Original, reproduced-defect and enhanced integration tests |
| `results/test_run.log`, `results/test_results.xml` | Actual pytest output and JUnit evidence |
| `results/enhanced/` | Complete v2 search, nested predictions, uncertainty and daily errors |
| `.github/workflows/tests.yml` | Automated tests and notebook execution on GitHub |

## Verification and limitations

The real-data experiments and all 49 automated tests passed in the delivery environment. All 13 notebook code cells were executed sequentially through IPython in a fresh Python process, with genuine outputs captured and notebook-schema validation. The full expanded searches were repeated by the notebook. Actual logs and JUnit XML are included. This host rejects Jupyter kernel socket setup, so ordinary Jupyter kernel startup could not be verified here. The standard kernel runner is included for your machine and CI. No Windows/macOS test run or GitHub Actions run is claimed.

Tests cover dataset integrity, malformed inputs, target exclusions, fold separation, past-only lag construction at multiple cutoffs, training-only scaling, every candidate family on real data, independently recomputed metrics, nested predictions, CLI consistency, source hashes, overwrite protection and selection rules. See CHANGELOG.md for the three initially failing regression cases. Small arrays are used only for metric-unit tests; fitted experiments use the actual observations.

Only one dwelling and part of one year are represented. Strong residual dependence, high peak-demand error and correlated predictors restrict interpretation. Coefficients are associations, not causal effects or promises of energy savings. Further tuning requires a new final holdout. The manuscript is prepared for author review and is not peer reviewed or accepted by a journal.

## GitHub upload

Create an empty repository, then run these commands from the extracted `WattWise` directory, replacing the URL with your own:

```bash
git init
git add .
git commit -m "Add WattWise research project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/WattWise.git
git push -u origin main
```

You can read the notebook and Markdown directly on GitHub. GitHub Pages does not execute Python notebooks. The workflow runs tests and notebook execution after a push or pull request; check its first run before advertising a passing CI badge.

## Attribution and licensing

Software and project documentation: MIT License, copyright 2026 Swayam Shakya. The dataset is separately **CC BY 4.0** and remains credited to Luis Candanedo. See [DATA_LICENSE.md](DATA_LICENSE.md). No raw data were modified. See [CITATION.cff](CITATION.cff) for project citation metadata.

1. Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G
2. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81-97. https://doi.org/10.1016/j.enbuild.2017.01.083
3. [Original research repository](https://github.com/LuisM78/Appliances-energy-prediction-data)
4. [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) and [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Additional method references: [scikit-learn lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) and [SplineTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html). The project implements its own experiment protocol; the examples are methodological references, not sources of result claims.
