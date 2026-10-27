"""Estimate same-interval energy with stable v1 ridge or the explicitly experimental v2 recipe."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.energy import make_features,load_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.inference import validate_conditions,fit_selected


def predict_csv(source,destination,experimental_v2=False):
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError('Output path must differ from input path')
    rows=validate_conditions(pd.read_csv(source))
    if experimental_v2:
        model,name,train=fit_selected('environment')
    else:
        df=load_data();cut=int(len(df)*.8);train=make_features(df.iloc[:cut])
        model=make_pipeline(StandardScaler(),Ridge(alpha=1000.)).fit(train,df.Appliances.iloc[:cut])
        name='Ridge calendar (v1 reference)'
    x=make_features(rows)
    raw=model.predict(x)
    result=pd.DataFrame({'date':rows.date,'prediction_raw_Wh':raw,
                         'prediction_nonnegative_Wh':np.maximum(raw,0),
                         'outside_training_range':((x<train.min()) | (x>train.max())).any(axis=1),
                         'model':name})
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('output')
    parser.add_argument('--experimental-v2',action='store_true',help='Use the v2 CV-selected environment recipe; it did not improve the retrospective holdout')
    args=parser.parse_args()
    print(predict_csv(args.input,args.output,args.experimental_v2).to_string(index=False))
