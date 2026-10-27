"""Validated inference using the selected v2 recipe, without pickle files."""
import json
import numpy as np
import pandas as pd
from .energy import ROOT, ENV, load_data, make_features
from .enhanced import recipes, history_features


def selected_model(task):
    selection=json.loads((ROOT/f'results/enhanced/{task}_selection.json').read_text())
    allowed=recipes(task)
    if selection['model'] not in allowed:
        raise ValueError('Unknown saved model recipe; rerun enhancements')
    model,grid=allowed[selection['model']]
    params=selection['parameters']
    if set(params)!=set(grid) or any(params[k] not in grid[k] for k in params):
        raise ValueError('Saved parameters do not match the declared search')
    return model.set_params(**params),selection['model']


def fit_selected(task):
    df=load_data();cut=int(len(df)*.8)
    x=make_features(df.iloc[:cut]) if task=='environment' else history_features(df.iloc[:cut])
    valid=x.notna().all(axis=1)
    model,name=selected_model(task)
    model.fit(x.loc[valid],df.Appliances.iloc[:cut].loc[valid])
    return model,name,x.loc[valid]


def validate_conditions(rows):
    rows=rows.copy()
    missing=set(ENV+['date'])-set(rows)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if rows.empty:raise ValueError('At least one condition row is required')
    rows.date=pd.to_datetime(rows.date,format='ISO8601',errors='raise')
    if rows.date.isna().any():raise ValueError('Missing dates are not allowed')
    try:
        rows[ENV]=rows[ENV].apply(pd.to_numeric,errors='raise')
    except (TypeError,ValueError) as error:
        raise ValueError('Conditions must contain numeric measurements') from error
    if not np.isfinite(rows[ENV].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite or missing measurements are not allowed')
    return rows


def forecast_next(history):
    """Forecast immediately after >=144 ordered ten-minute measured readings."""
    history=history.copy().reset_index(drop=True)
    if not {'date','Appliances'} <= set(history):raise ValueError('History needs date and Appliances columns')
    if len(history)<144:raise ValueError('At least 144 readings (24 hours) are required')
    history.date=pd.to_datetime(history.date,format='ISO8601',errors='raise')
    if history.date.isna().any() or not history.date.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
        raise ValueError('History must have ordered, unique, uninterrupted ten-minute timestamps')
    history.Appliances=pd.to_numeric(history.Appliances,errors='raise')
    if not np.isfinite(history.Appliances).all() or (history.Appliances<0).any():
        raise ValueError('History energy must be finite and nonnegative')
    # The current target is intentionally unknown, and must never enter features.
    date=history.date.iloc[-1]+pd.Timedelta(minutes=10)
    extended=pd.concat([history[['date','Appliances']],pd.DataFrame({'date':[date],'Appliances':[np.nan]})],ignore_index=True)
    x=history_features(extended).iloc[[-1]]
    model,name,_=fit_selected('forecast')
    raw=float(model.predict(x)[0])
    return pd.DataFrame({'date':[date],'prediction_raw_Wh':[raw],
                         'prediction_nonnegative_Wh':[max(raw,0)],'model':[name]})
