"""Restore the exact UCI dataset, checking its recorded SHA-256."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip'

def main():
    with urllib.request.urlopen(URL,timeout=90) as response:
        payload=response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        content=archive.read('energydata_complete.csv')
    expected=json.loads((ROOT/'data/provenance.json').read_text())['sha256']
    if hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError('Source content changed; inspect it before updating the benchmark')
    target=ROOT/'data/raw/energydata_complete.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content)
    print(target)

if __name__=='__main__': main()
