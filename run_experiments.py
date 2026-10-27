"""Run from any working directory: python /path/to/run_experiments.py."""
from src.energy import load_data, fit_experiments, rolling_forecast, diagnostics, make_plots

def main():
    df = load_data()
    scores, predictions, models, manifest = fit_experiments(df)
    forecast, _ = rolling_forecast(df)
    diagnostics(df, predictions, models, manifest)
    make_plots(df, scores, predictions, manifest)
    print(scores.to_string(index=False))
    print('\nSeparate rolling one-step forecast:\n', forecast.to_string(index=False))
    print('\nSelected by training CV:', manifest['selected_by_cv'])

if __name__ == '__main__':
    main()
