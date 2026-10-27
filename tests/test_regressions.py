"""Regression checks added during the second audit."""
import numpy as np
import pandas as pd
import pytest
from src.energy import load_data, rolling_forecast


def test_empty_csv_is_rejected(tmp_path):
    path=tmp_path/'empty.csv'
    load_data().iloc[:0].to_csv(path,index=False)
    with pytest.raises(ValueError,match='at least'):
        load_data(path)


def test_source_hash_tracks_actual_file(tmp_path):
    path=tmp_path/'subset.csv'
    load_data().iloc[:1000].to_csv(path,index=False)
    frame=load_data(path)
    import hashlib
    assert frame.attrs['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()


def test_forecast_ignores_arbitrary_index_labels(tmp_path):
    df=load_data().iloc[:3000].copy()
    left=tmp_path/'a';right=tmp_path/'b'
    left.mkdir();right.mkdir()
    a,_=rolling_forecast(df,out=left)
    df.index=np.arange(len(df))+50000
    b,_=rolling_forecast(df,out=right)
    pd.testing.assert_frame_equal(a,b)
