"""Forecast the next interval using a CSV containing date and Appliances history."""
from pathlib import Path
import argparse
import pandas as pd
from src.inference import forecast_next


def forecast_csv(source,destination):
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError('Output path must differ from input path')
    result=forecast_next(pd.read_csv(source))
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('output')
    args=parser.parse_args()
    print(forecast_csv(args.input,args.output).to_string(index=False))
