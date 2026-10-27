"""Reproducible experiments for WattWise. Author: Swayam Shakya."""
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/energydata_complete.csv'
SEED = 42
ENV = [v for i in range(1, 10) for v in (f'T{i}', f'RH_{i}')] + ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
CAL = ['hour_sin', 'hour_cos', 'week_sin', 'week_cos', 'weekend']


def load_data(path=DATA):
    """Reject invalid observations instead of silently repairing a benchmark."""
    df = pd.read_csv(path)
    if len(df) < 2:
        raise ValueError('Dataset must contain at least two observations')
    required = ['date', 'Appliances', 'lights', 'rv1', 'rv2'] + ENV
    if set(required) - set(df):
        raise ValueError('Missing required dataset columns')
    df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='raise')
    if df[required].isna().any().any():
        raise ValueError('Missing values found')
    if not np.isfinite(df[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite measurements found')
    if not df.date.is_monotonic_increasing or df.date.duplicated().any():
        raise ValueError('Timestamps must be unique and ordered')
    if not df.date.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
        raise ValueError('Expected uninterrupted ten-minute intervals')
    if (df.Appliances < 0).any():
        raise ValueError('Energy cannot be negative')
    df.attrs['source_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return df


def calendar_features(dates):
    dates = pd.to_datetime(dates)
    hour = dates.dt.hour + dates.dt.minute / 60
    week = dates.dt.dayofweek + hour / 24
    return pd.DataFrame({'hour_sin': np.sin(2*np.pi*hour/24),
                         'hour_cos': np.cos(2*np.pi*hour/24),
                         'week_sin': np.sin(2*np.pi*week/7),
                         'week_cos': np.cos(2*np.pi*week/7),
                         'weekend': (dates.dt.dayofweek >= 5).astype(int)}, index=dates.index)


def make_features(df, calendar=True):
    x = df[ENV].copy()
    return pd.concat([x, calendar_features(df.date)], axis=1) if calendar else x


def forecast_features(df):
    """At target time t, use readings through t-1 only (rolling one-step use)."""
    x = calendar_features(df.date)
    for lag in [1, 6, 144]:
        x[f'lag_{lag}'] = df.Appliances.shift(lag)
    x['past_hour_mean'] = df.Appliances.shift(1).rolling(6).mean()
    return x


def metrics(y, prediction):
    return {'MAE_Wh': float(mean_absolute_error(y, prediction)),
            'RMSE_Wh': float(np.sqrt(mean_squared_error(y, prediction))),
            'R2': float(r2_score(y, prediction)),
            'negative_predictions': int((np.asarray(prediction) < 0).sum())}


def temporal_cv():
    return TimeSeriesSplit(n_splits=4, gap=144)


def candidates():
    linear = lambda model: make_pipeline(StandardScaler(), model)
    return {
        'Mean baseline': (DummyRegressor(strategy='mean'), {}, False),
        'Median baseline': (DummyRegressor(strategy='median'), {}, False),
        'MLR environment': (linear(LinearRegression()), {}, False),
        'MLR calendar': (linear(LinearRegression()), {}, True),
        'Ridge calendar': (linear(Ridge()), {'ridge__alpha': [0.1, 10., 1000.]}, True),
        'Gradient boosting': (HistGradientBoostingRegressor(max_iter=150, early_stopping=False, random_state=SEED),
                              {'max_leaf_nodes': [7, 15], 'l2_regularization': [1., 10.]}, True)
    }


def fit_experiments(df, out=ROOT/'results'):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cut = int(len(df)*0.8)
    rows, folds, fitted, predictions = [], [], {}, pd.DataFrame({'date': df.date.iloc[cut:], 'actual': df.Appliances.iloc[cut:]})
    for name, (estimator, grid, use_calendar) in candidates().items():
        x = make_features(df, use_calendar); y = df.Appliances
        search = GridSearchCV(estimator, grid, cv=temporal_cv(), scoring='neg_root_mean_squared_error', n_jobs=1, refit=True, error_score='raise')
        start = time.perf_counter()
        search.fit(x.iloc[:cut], y.iloc[:cut])
        pred = search.predict(x.iloc[cut:])
        rows.append({'model': name, 'CV_RMSE_Wh': -search.best_score_, **metrics(y.iloc[cut:], pred), 'fit_seconds': time.perf_counter()-start, 'parameters': json.dumps(search.best_params_)})
        for i, (tr, va) in enumerate(temporal_cv().split(x.iloc[:cut])):
            folds.append({'model': name, 'fold': i+1, 'train_end': str(df.date.iloc[tr[-1]]), 'validation_start': str(df.date.iloc[va[0]]), 'validation_end': str(df.date.iloc[va[-1]]), 'RMSE_Wh': -search.cv_results_[f'split{i}_test_score'][search.best_index_]})
        fitted[name] = search.best_estimator_
        predictions[name] = pred
    scores = pd.DataFrame(rows)
    selected = scores.loc[scores.CV_RMSE_Wh.idxmin(), 'model']
    scores.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'cv_folds.csv', index=False)
    predictions.to_csv(out/'predictions.csv', index=False)
    manifest = {'seed': SEED, 'data_sha256': df.attrs.get('source_sha256'), 'frame_sha256': hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(), 'rows': len(df), 'training_rows': cut, 'test_rows': len(df)-cut,
                'train_start': str(df.date.iloc[0]), 'train_end': str(df.date.iloc[cut-1]), 'test_start': str(df.date.iloc[cut]), 'test_end': str(df.date.iloc[-1]),
                'selected_by_cv': selected, 'cv_splits': 4, 'cv_gap_rows': 144,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return scores, predictions, fitted, manifest


def rolling_forecast(df, out=ROOT/'results'):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    cut = int(len(df)*0.8)
    x = forecast_features(df)
    valid = x.notna().all(axis=1)
    train = valid & (df.index < cut); test = valid & (df.index >= cut)
    search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': [0.1, 10., 1000.]}, cv=temporal_cv(), scoring='neg_root_mean_squared_error', error_score='raise')
    search.fit(x.loc[train], df.Appliances.loc[train])
    p = pd.DataFrame({'date': df.date.loc[test], 'actual': df.Appliances.loc[test], 'Persistence': x.loc[test, 'lag_1'], 'Previous day': x.loc[test, 'lag_144'], 'Lagged ridge': search.predict(x.loc[test])})
    rows = [{'model': name, **metrics(p.actual, p[name])} for name in ['Persistence', 'Previous day', 'Lagged ridge']]
    pd.DataFrame(rows).to_csv(Path(out)/'forecast_metrics.csv', index=False)
    p.to_csv(Path(out)/'forecast_predictions.csv', index=False)
    (Path(out)/'forecast_config.json').write_text(json.dumps({'best_parameters': search.best_params_, 'CV_RMSE_Wh': -search.best_score_, 'training_rows': int(train.sum()), 'features': list(x)}, indent=2))
    return pd.DataFrame(rows), p


def block_interval(y, a, b, block=144, repeats=1000):
    """Paired circular block bootstrap: RMSE(a)-RMSE(b), conditional on fits."""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(SEED); n = len(y); differences = []
    for _ in range(repeats):
        starts = rng.integers(0, n, size=int(np.ceil(n/block)))
        ids = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        differences.append(np.sqrt(np.mean((y[ids]-a[ids])**2))-np.sqrt(np.mean((y[ids]-b[ids])**2)))
    return np.percentile(differences, [2.5, 97.5]).tolist()


def diagnostics(df, predictions, models, manifest, out=ROOT/'results'):
    out = Path(out); cut = manifest['training_rows']
    x = make_features(df); z = models['MLR calendar'].named_steps['standardscaler'].transform(x.iloc[:cut])
    ols = models['MLR calendar'].named_steps['linearregression']
    pd.DataFrame({'feature': x.columns, 'coefficient_Wh_per_training_SD': ols.coef_}).to_csv(out/'coefficients.csv', index=False)
    vifs = []
    for j, feature in enumerate(x.columns):
        r2 = LinearRegression().fit(np.delete(z, j, axis=1), z[:, j]).score(np.delete(z, j, axis=1), z[:, j])
        vifs.append({'feature': feature, 'VIF': 1/max(1-r2, 1e-12)})
    pd.DataFrame(vifs).to_csv(out/'vif.csv', index=False)
    residual = predictions.actual-predictions['MLR calendar']
    threshold = df.Appliances.iloc[:cut].quantile(.9)
    peak = predictions.actual >= threshold
    summary = {'MLR_residual_lag1_correlation': float(residual.autocorr()), 'training_peak_threshold_Wh': float(threshold),
               'MLR_peak_MAE_Wh': float(residual[peak].abs().mean()), 'MLR_nonpeak_MAE_Wh': float(residual[~peak].abs().mean()),
               'test_peak_count': int(peak.sum()), 'condition_number_scaled_design': float(np.linalg.cond(np.column_stack([np.ones(len(z)), z]))),
               'MLR_calendar_minus_mean_RMSE_CI_95_Wh': block_interval(predictions.actual, predictions['MLR calendar'], predictions['Mean baseline']),
               'bootstrap_block_rows': 144, 'bootstrap_repeats': 1000}
    (out/'diagnostics.json').write_text(json.dumps(summary, indent=2))
    return summary


def make_plots(df, scores, pred, manifest, out=ROOT/'results/figures'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold'})
    cut=manifest['training_rows']; train=df.iloc[:cut]
    def save(name):
        plt.tight_layout(); plt.savefig(out/f'{name}.png', bbox_inches='tight'); plt.close()
    fig, ax=plt.subplots(figsize=(10,3.5))
    daily=df.set_index('date').Appliances.resample('D').mean()
    ax.plot(daily.index, daily, color='#007f86'); ax.axvline(df.date.iloc[cut], color='#d45936', ls='--', label='Test begins')
    ax.set(title='Daily mean appliance energy per 10-minute interval', ylabel='Wh / interval'); ax.legend(); save('timeline')
    fig, axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].hist(train.Appliances,bins=50,color='#007f86'); axes[0].set(title='Training target distribution',xlabel='Wh / interval',ylabel='Count')
    train.groupby(train.date.dt.hour).Appliances.mean().plot(ax=axes[1],color='#d45936'); axes[1].set(title='Training hourly profile',xlabel='Hour',ylabel='Mean Wh / interval'); save('eda')
    fig, ax=plt.subplots(figsize=(8,7))
    corr=make_features(train,False).corr(); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=7); ax.set_yticks(range(len(corr)),corr.columns,fontsize=7)
    ax.set_title('Training predictor correlations'); fig.colorbar(im,ax=ax,shrink=.8); save('correlations')
    fig, ax=plt.subplots(figsize=(9,4))
    scores.plot.barh(x='model',y=['CV_RMSE_Wh','RMSE_Wh'],ax=ax,color=['#007f86','#d45936']); ax.set(xlabel='RMSE (Wh)',ylabel='',title='Chronological evaluation'); save('comparison')
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    residual=pred.actual-pred['MLR calendar']
    axes[0].scatter(pred['MLR calendar'],residual,s=4,alpha=.25,color='#007f86'); axes[0].axhline(0,color='black',lw=1)
    axes[0].set(xlabel='Predicted Wh',ylabel='Actual - predicted Wh',title='MLR residuals')
    axes[1].plot(range(1,145),[residual.autocorr(lag=i) for i in range(1,145)],color='#d45936'); axes[1].set(xlabel='Lag (10-minute steps)',ylabel='Residual autocorrelation',title='Remaining temporal structure'); save('residuals')
    fig, ax=plt.subplots(figsize=(10,3.5)); window=pred.iloc[:288]
    ax.plot(pd.to_datetime(window.date),window.actual,label='Measured',lw=1,color='#243342')
    ax.plot(pd.to_datetime(window.date),window['MLR calendar'],label='MLR calendar',lw=1,color='#d45936')
    ax.set(title='First 48 hours of held-out predictions',ylabel='Wh / interval'); ax.legend(); fig.autofmt_xdate(); save('predictions')
