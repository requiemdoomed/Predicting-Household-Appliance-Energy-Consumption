"""Run all v2 searches and nested validation; prints actual measured progress."""
from src.energy import load_data
from src.enhanced import run_enhancements

if __name__=='__main__':
    run_enhancements(load_data())
