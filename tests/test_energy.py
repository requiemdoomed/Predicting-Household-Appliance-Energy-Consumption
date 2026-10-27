"""Tests of real-data integrity, chronology, leakage boundaries and outputs."""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.energy import ROOT, load_data, make_features, forecast_features, temporal_cv, metrics, block_interval

@pytest.fixture(scope='module')
def df():
    return load_data()


def test_real_dataset_shape_and_units(df):
    assert df.shape == (19735,29)
    assert df.Appliances.min() == 10
    assert df.date.iloc[0] == pd.Timestamp('2016-01-11 17:00:00')


def test_no_target_or_random_controls(df):
    x = make_features(df)
    assert not {'Appliances','lights','rv1','rv2','date'} & set(x.columns)
    assert x.shape == (19735,29)


def test_time_folds_respect_gap_and_holdout(df):
    cut=int(len(df)*.8)
    for tr, va in temporal_cv().split(df.iloc[:cut]):
        assert tr[-1] + 145 == va[0]
        assert va[-1] < cut
        assert df.date.iloc[tr[-1]] < df.date.iloc[va[0]]


def test_future_target_does_not_change_past_features(df):
    original = forecast_features(df)
    changed=df.copy(); changed.loc[1000:, 'Appliances'] += 10000
    pd.testing.assert_frame_equal(original.iloc[:1001], forecast_features(changed).iloc[:1001])
    assert original.loc[1000,'lag_1'] == df.loc[999,'Appliances']
    assert original.loc[1000,'past_hour_mean'] == df.loc[994:999,'Appliances'].mean()


def test_scaling_uses_training_only(df):
    x=make_features(df); cut=int(len(df)*.8)
    model=make_pipeline(StandardScaler(),LinearRegression()).fit(x.iloc[:cut],df.Appliances.iloc[:cut])
    np.testing.assert_allclose(model[0].mean_,x.iloc[:cut].mean(),atol=1e-10)
    assert not np.allclose(model[0].mean_,x.mean())


def test_metrics_known_answer():
    m=metrics([1,2,3],[2,2,2])
    assert m['MAE_Wh'] == pytest.approx(2/3)
    assert m['RMSE_Wh'] == pytest.approx(np.sqrt(2/3))
    assert m['R2'] == 0


@pytest.mark.parametrize('problem',['duplicate','missing','negative','infinite','gap'])
def test_reject_bad_input(df,tmp_path,problem):
    x=df.iloc[:20].copy()
    if problem=='duplicate': x.loc[1,'date']=x.loc[0,'date']
    if problem=='missing': x.loc[0,'T1']=np.nan
    if problem=='negative': x.loc[0,'Appliances']=-1
    if problem=='infinite': x.loc[0,'T1']=np.inf
    if problem=='gap': x=x.drop(index=1)
    p=tmp_path/'bad.csv';x.to_csv(p,index=False)
    with pytest.raises(ValueError): load_data(p)


def test_saved_scores_recompute():
    p=pd.read_csv(ROOT/'results/predictions.csv')
    s=pd.read_csv(ROOT/'results/metrics.csv')
    for _,row in s.iterrows():
        for name,value in metrics(p.actual,p[row.model]).items():
            assert row[name] == pytest.approx(value)


def test_bootstrap_identity(df):
    y=df.Appliances.iloc[:500].to_numpy()
    assert block_interval(y,y+1,y+1,repeats=30) == [0,0]


def test_selection_uses_cv():
    s=pd.read_csv(ROOT/'results/metrics.csv')
    m=json.loads((ROOT/'results/manifest.json').read_text())
    assert m['selected_by_cv'] == s.loc[s.CV_RMSE_Wh.idxmin(),'model']
