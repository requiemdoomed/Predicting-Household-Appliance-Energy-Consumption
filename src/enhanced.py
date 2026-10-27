"""Version 2: bounded searches and nested temporal validation, without test tuning."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from .energy import ROOT, SEED, make_features, forecast_features, metrics, block_interval


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """No fitted parameters: use the last observed target for every forecast."""
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return X['lag_1'].to_numpy()


def history_features(df):
    """All windows end at t-1; targets at t and later cannot affect row t."""
    x=forecast_features(df)
    for lag in [2, 3, 12, 36]:
        x[f'lag_{lag}']=df.Appliances.shift(lag)
    past=df.Appliances.shift(1)
    for window in [6,36,144]:
        roll=past.rolling(window)
        x[f'mean_{window}']=roll.mean()
        x[f'std_{window}']=roll.std(ddof=0)
        x[f'min_{window}']=roll.min()
        x[f'max_{window}']=roll.max()
    # Avoid duplicating the original six-reading mean.
    x=x.drop(columns='mean_6')
    x['last_change']=x.lag_1-x.lag_2
    return x


def recipes(task):
    linear=lambda model: make_pipeline(StandardScaler(),model)
    common={'Mean baseline':(DummyRegressor(strategy='mean'),{}),
            'Median baseline':(DummyRegressor(strategy='median'),{})}
    if task=='environment':
        return {**common,
            'MLR calendar':(linear(LinearRegression()),{}),
            'Ridge expanded':(linear(Ridge()),{'ridge__alpha':[100.,1000.,10000.,100000.]}),
            'Additive splines':(make_pipeline(SplineTransformer(include_bias=False,extrapolation='constant'),StandardScaler(),Ridge()),
                                {'splinetransformer__n_knots':[3,5],'ridge__alpha':[100.,1000.]}),
            'Extra trees':(ExtraTreesRegressor(n_estimators=100,random_state=SEED,n_jobs=1,max_features=.8),{'min_samples_leaf':[20,60]})}
    if task=='forecast':
        return {**common,
            'Persistence':(PersistenceRegressor(),{}),
            'History ridge':(linear(Ridge()),{'ridge__alpha':[.1,10.,1000.,10000.]}),
            'History boosting':(HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,early_stopping=False,random_state=SEED),
                                {'max_leaf_nodes':[7,15], 'l2_regularization':[10.,100.], 'loss':['squared_error','poisson']}),
            'History extra trees':(ExtraTreesRegressor(n_estimators=100,random_state=SEED,n_jobs=1,max_features=.8),{'min_samples_leaf':[5,20]})}
    raise ValueError('Unknown task')


def select_recipe(X,y,task,splits=4):
    """Select using only the arrays passed in; holdout data are not accepted."""
    fitted={}; trials=[]; best=[]
    cv=TimeSeriesSplit(n_splits=splits,gap=144)
    for name,(model,grid) in recipes(task).items():
        search=GridSearchCV(model,grid,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=1,error_score='raise',return_train_score=True)
        search.fit(X,y)
        fitted[name]=search.best_estimator_
        best.append({'model':name,'CV_RMSE_Wh':float(-search.best_score_),'parameters':search.best_params_})
        for j,params in enumerate(search.cv_results_['params']):
            row={'model':name,'parameters':json.dumps(params,sort_keys=True),'mean_CV_RMSE_Wh':float(-search.cv_results_['mean_test_score'][j])}
            for i in range(splits):
                row[f'fold_{i+1}_RMSE_Wh']=float(-search.cv_results_[f'split{i}_test_score'][j])
                row[f'fold_{i+1}_train_RMSE_Wh']=float(-search.cv_results_[f'split{i}_train_score'][j])
            trials.append(row)
    selected=min(best,key=lambda r:r['CV_RMSE_Wh'])
    return selected,fitted,pd.DataFrame(trials),best


def nested_validation(X,y,dates,task):
    """Outer folds evaluate the entire inner model-family selection procedure."""
    rows=[]; predictions=[]
    for fold,(tr,va) in enumerate(TimeSeriesSplit(n_splits=3,gap=144).split(X),1):
        start=time.perf_counter()
        selected,models,_,_=select_recipe(X.iloc[tr],y.iloc[tr],task,splits=3)
        name=selected['model']; pred=models[name].predict(X.iloc[va])
        baseline=X.iloc[va].lag_1.to_numpy() if task=='forecast' else np.full(len(va),y.iloc[tr].mean())
        rows.append({'fold':fold,'selected_model':name,'parameters':json.dumps(selected['parameters']),
                     'train_end':str(dates.iloc[tr[-1]]),'validation_start':str(dates.iloc[va[0]]),'validation_end':str(dates.iloc[va[-1]]),
                     'train_rows':len(tr),'validation_rows':len(va),'gap_rows':144,**metrics(y.iloc[va],pred),
                     'baseline_RMSE_Wh':metrics(y.iloc[va],baseline)['RMSE_Wh'],'seconds':time.perf_counter()-start})
        predictions.append(pd.DataFrame({'date':dates.iloc[va].to_numpy(),'fold':fold,'actual':y.iloc[va].to_numpy(),'prediction':pred,'baseline':baseline}))
        print(f'{task} nested fold {fold}: {name}, RMSE={rows[-1]["RMSE_Wh"]:.3f}',flush=True)
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)


def run_enhancements(df,out=ROOT/'results/enhanced'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=df.reset_index(drop=True);cut=int(len(df)*.8)
    config={'version':'2.0.0','seed':SEED,'criterion':'mean expanding-fold RMSE','development_rows':cut,
            'holdout_status':'Previously inspected in v1; v2 comparisons are exploratory, not a new untouched test.',
            'inner_folds_final':4,'outer_folds':3,'inner_folds_nested':3,'gap_rows':144,
            'searches':{t:{name:{'estimator':repr(model),'grid':grid,'configurations':len(list(ParameterGrid(grid)))} for name,(model,grid) in recipes(t).items()} for t in ['environment','forecast']}}
    (out/'protocol.json').write_text(json.dumps(config,indent=2))
    # Fit and select both tasks before computing any holdout metrics.
    fitted_tasks={}
    for task in ['environment','forecast']:
        X=make_features(df) if task=='environment' else history_features(df)
        valid=X.notna().all(axis=1);tr=valid & (df.index<cut);te=valid & (df.index>=cut)
        Xtrain=X.loc[tr];ytrain=df.Appliances.loc[tr];dates=df.date.loc[tr]
        selected,models,trials,best=select_recipe(Xtrain,ytrain,task)
        trials.to_csv(out/f'{task}_search.csv',index=False)
        (out/f'{task}_selection.json').write_text(json.dumps(selected,indent=2))
        print(f'{task}: final CV selected {selected["model"]}, RMSE={selected["CV_RMSE_Wh"]:.3f}',flush=True)
        nested,nested_predictions=nested_validation(Xtrain,ytrain,dates,task)
        nested.to_csv(out/f'{task}_nested.csv',index=False)
        nested_predictions.to_csv(out/f'{task}_nested_predictions.csv',index=False)
        fitted_tasks[task]=(X,te,selected,models,best)
    for task,(X,te,selected,models,best) in fitted_tasks.items():
        p=pd.DataFrame({'date':df.date.loc[te],'actual':df.Appliances.loc[te]});rows=[]
        for recipe in best:
            name=recipe['model'];p[name]=models[name].predict(X.loc[te])
            rows.append({'model':name,'CV_RMSE_Wh':recipe['CV_RMSE_Wh'],**metrics(p.actual,p[name]),'selected_by_CV':name==selected['model'],'parameters':json.dumps(recipe['parameters'])})
        scores=pd.DataFrame(rows)
        scores.to_csv(out/f'{task}_metrics.csv',index=False);p.to_csv(out/f'{task}_predictions.csv',index=False)
        baseline='Persistence' if task=='forecast' else 'Mean baseline'
        intervals=[]
        for block in [36,144,288]:
            low,high=block_interval(p.actual,p[selected['model']],p[baseline],block=block,repeats=1000)
            intervals.append({'block_rows':block,'low_Wh':low,'high_Wh':high})
        (out/f'{task}_uncertainty.json').write_text(json.dumps({'comparison':f'{selected["model"]} minus {baseline}','interpretation':'Exploratory paired RMSE difference, conditional on fitted models; not selection-adjusted','intervals_95_percent':intervals},indent=2))
        # Absolute errors by day expose variation hidden by one global average.
        days=pd.DataFrame({'date':pd.to_datetime(p.date).dt.date,'absolute_error':abs(p.actual-p[selected['model']]),'baseline_absolute_error':abs(p.actual-p[baseline])})
        days.groupby('date').mean().to_csv(out/f'{task}_daily_errors.csv')
        print('\n'+task+' retrospective holdout:\n'+scores.to_string(index=False),flush=True)
    make_enhanced_plots(out)
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    (out/'artifact_hashes.json').write_text(json.dumps(files,indent=2))
    return {task:pd.read_csv(out/f'{task}_metrics.csv') for task in fitted_tasks}


def make_enhanced_plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,task in zip(axes,['environment','forecast']):
        s=pd.read_csv(Path(out)/f'{task}_metrics.csv')
        ax.barh(s.model,s.RMSE_Wh,color=['#007f86' if x else '#a6b8c5' for x in s.selected_by_CV])
        ax.set(title=f'{task.title()}: retrospective holdout',xlabel='RMSE (Wh)')
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(Path(out)/'comparison.png',dpi=150);plt.close(fig)
