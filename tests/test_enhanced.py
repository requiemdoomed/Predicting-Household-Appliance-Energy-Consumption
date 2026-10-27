"""Real-data invariance, provenance, selection and integration checks."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.energy import ROOT, load_data, metrics, make_features
from src.enhanced import history_features, recipes, PersistenceRegressor


@pytest.fixture(scope='module')
def data():return load_data()


@pytest.mark.parametrize('boundary',[144,1000,15788,19000])
def test_every_history_feature_ignores_current_and_future_targets(data,boundary):
    before=history_features(data)
    altered=data.copy();altered.loc[boundary:,'Appliances']+=12345
    pd.testing.assert_frame_equal(before.iloc[:boundary+1],history_features(altered).iloc[:boundary+1])


def test_history_values_are_hand_checked_against_real_readings(data):
    x=history_features(data);t=1000
    for lag in [1,2,3,6,12,36,144]:
        assert x.loc[t,f'lag_{lag}']==data.Appliances.iloc[t-lag]
    for window in [6,36,144]:
        readings=data.Appliances.iloc[t-window:t]
        mean_name='past_hour_mean' if window==6 else f'mean_{window}'
        assert x.loc[t,mean_name]==pytest.approx(readings.mean())
        assert x.loc[t,f'std_{window}']==pytest.approx(readings.std(ddof=0))
        assert x.loc[t,f'min_{window}']==readings.min()
        assert x.loc[t,f'max_{window}']==readings.max()


def test_all_recipe_predictions_are_finite_on_real_data(data):
    for task in ['environment','forecast']:
        x=(make_features(data) if task=='environment' else history_features(data)).iloc[144:1000]
        y=data.Appliances.loc[x.index]
        for name,(model,_) in recipes(task).items():
            model.fit(x.iloc[:600],y.iloc[:600])
            pred=model.predict(x.iloc[600:])
            assert len(pred)==len(x)-600,name
            assert np.isfinite(pred).all(),name


def test_persistence_does_not_fit_targets(data):
    x=history_features(data).iloc[144:300]
    model=PersistenceRegressor().fit(x,data.Appliances.iloc[144:300]*1000)
    np.testing.assert_array_equal(model.predict(x),x.lag_1)


@pytest.mark.parametrize('task',['environment','forecast'])
def test_enhanced_selection_and_saved_scores(task):
    root=ROOT/'results/enhanced'
    selection=json.loads((root/f'{task}_selection.json').read_text())
    search=pd.read_csv(root/f'{task}_search.csv')
    winner=search.loc[search.mean_CV_RMSE_Wh.idxmin()]
    assert winner.model==selection['model']
    assert winner.mean_CV_RMSE_Wh==pytest.approx(selection['CV_RMSE_Wh'])
    scores=pd.read_csv(root/f'{task}_metrics.csv');pred=pd.read_csv(root/f'{task}_predictions.csv')
    for _,row in scores.iterrows():
        for k,v in metrics(pred.actual,pred[row.model]).items():assert row[k]==pytest.approx(v)
    assert scores.selected_by_CV.sum()==1
    assert scores.loc[scores.selected_by_CV,'model'].iloc[0]==selection['model']


@pytest.mark.parametrize('task',['environment','forecast'])
def test_nested_predictions_and_boundaries(task):
    root=ROOT/'results/enhanced'
    scores=pd.read_csv(root/f'{task}_nested.csv');pred=pd.read_csv(root/f'{task}_nested_predictions.csv')
    for _,row in scores.iterrows():
        assert pd.Timestamp(row.validation_start)-pd.Timestamp(row.train_end)==pd.Timedelta(minutes=1450)
        subset=pred.loc[pred.fold==row.fold]
        assert len(subset)==row.validation_rows
        assert metrics(subset.actual,subset.prediction)['RMSE_Wh']==pytest.approx(row.RMSE_Wh)
        assert pd.Timestamp(row.validation_end)<pd.Timestamp(json.loads((ROOT/'results/manifest.json').read_text())['test_start'])


@pytest.mark.parametrize('issue',['missing_column','empty','missing_date','bad_date','missing_value','infinite','text'])
def test_condition_validation_rejects_malformed_input(data,issue):
    from src.inference import validate_conditions
    x=data.iloc[:2].copy()
    if issue=='missing_column':x=x.drop(columns='T1')
    elif issue=='empty':x=x.iloc[:0]
    elif issue=='missing_date':x.loc[0,'date']=pd.NaT
    elif issue=='bad_date':x['date']=['invalid','invalid']
    elif issue=='missing_value':x.loc[0,'T1']=np.nan
    elif issue=='infinite':x.loc[0,'T1']=np.inf
    elif issue=='text':x['T1']=['bad','bad']
    with pytest.raises(ValueError):validate_conditions(x)


@pytest.mark.parametrize('issue',['short','gap','duplicate','negative','infinite','missing_date'])
def test_forecast_history_validation(data,issue):
    from src.inference import forecast_next
    x=data.iloc[-144:][['date','Appliances']].reset_index(drop=True)
    if issue=='short':x=x.iloc[:143]
    elif issue=='gap':x.loc[0,'date']-=pd.Timedelta(minutes=10)
    elif issue=='duplicate':x.loc[1,'date']=x.loc[0,'date']
    elif issue=='negative':x.loc[0,'Appliances']=-1
    elif issue=='infinite':x['Appliances']=np.inf
    elif issue=='missing_date':x.loc[0,'date']=pd.NaT
    with pytest.raises(ValueError):forecast_next(x)


def test_next_forecast_matches_last_saved_test_prediction(data):
    from src.inference import forecast_next
    history=data.iloc[-145:-1][['date','Appliances']]
    result=forecast_next(history)
    p=pd.read_csv(ROOT/'results/enhanced/forecast_predictions.csv')
    name=json.loads((ROOT/'results/enhanced/forecast_selection.json').read_text())['model']
    assert pd.Timestamp(result.date.iloc[0])==pd.Timestamp(p.date.iloc[-1])
    assert result.prediction_raw_Wh.iloc[0]==pytest.approx(p[name].iloc[-1])


def test_condition_cli_matches_saved_predictions(tmp_path):
    output=tmp_path/'output.csv'
    result=subprocess.run([sys.executable,str(ROOT/'predict.py'),str(ROOT/'data/example_conditions.csv'),str(output),'--experimental-v2'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    a=pd.read_csv(output);b=pd.read_csv(ROOT/'results/enhanced/environment_predictions.csv').iloc[-3:]
    name=json.loads((ROOT/'results/enhanced/environment_selection.json').read_text())['model']
    np.testing.assert_allclose(a.prediction_raw_Wh,b[name])


def test_inference_training_ignores_holdout_targets(data,monkeypatch):
    import src.inference as inference
    model,_,x=inference.fit_selected('environment')
    changed=data.copy();changed.loc[int(len(changed)*.8):,'Appliances']+=10000
    monkeypatch.setattr(inference,'load_data',lambda:changed)
    other,_,_=inference.fit_selected('environment')
    np.testing.assert_allclose(model.predict(x.iloc[:10]),other.predict(x.iloc[:10]),rtol=0,atol=0)


def test_prediction_cannot_overwrite_input(tmp_path):
    from predict import predict_csv
    from forecast import forecast_csv
    p=tmp_path/'input.csv';p.write_text('sentinel')
    for function in [predict_csv,forecast_csv]:
        with pytest.raises(ValueError,match='differ'):function(p,p)
    assert p.read_text()=='sentinel'


def test_data_provenance_checksum():
    import hashlib
    recorded=json.loads((ROOT/'data/provenance.json').read_text())['sha256']
    assert recorded==hashlib.sha256((ROOT/'data/raw/energydata_complete.csv').read_bytes()).hexdigest()


def test_default_conditions_retain_reference_ridge(tmp_path):
    from predict import predict_csv
    result=predict_csv(ROOT/'data/example_conditions.csv',tmp_path/'result.csv')
    original=pd.read_csv(ROOT/'results/predictions.csv').iloc[-3:]
    np.testing.assert_allclose(result.prediction_raw_Wh,original['Ridge calendar'])


def test_forecast_cli_matches_function(tmp_path):
    from src.inference import forecast_next
    output=tmp_path/'future.csv'
    result=subprocess.run([sys.executable,str(ROOT/'forecast.py'),str(ROOT/'data/example_history.csv'),str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    expected=forecast_next(pd.read_csv(ROOT/'data/example_history.csv'))
    assert pd.read_csv(output).prediction_raw_Wh.iloc[0]==pytest.approx(expected.prediction_raw_Wh.iloc[0])


def test_selected_recipe_rejects_undeclared_parameters(monkeypatch):
    import src.inference as inference
    real_loads=inference.json.loads
    def altered(text):
        value=real_loads(text)
        value['parameters']['undeclared']=10
        return value
    monkeypatch.setattr(inference.json,'loads',altered)
    with pytest.raises(ValueError,match='declared search'):inference.selected_model('forecast')
