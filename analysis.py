import numpy as np 
import astropy.units as u 
import h5py 
import matplotlib.pyplot as plt 
import pandas as pd
from astropy.constants import G
from pathlib import Path
import argparse

def load_data_file(file_path):
    """
    Checks the file extension and loads a .csv or .dat file into a Pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    file_extension = path.suffix.lower()
    
    if file_extension == '.csv':
        print(f"Processing CSV file: {path.name}")
        return pd.read_csv(path)
        
    elif file_extension == '.dat':
        print(f"Processing DAT file: {path.name}")
        return pd.read_csv(path, sep=None, engine='python')
        
    else:
        raise ValueError(f"Unsupported file type '{file_extension}'. Only .csv and .dat are allowed.")



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="Path to the cluster_output_<branchid>_<snapshot>.csv file")
    args = parser.parse_args()
    df = load_data_file(args.input_path)
    print(df.columns)

if __name__ == "__main__":
    main()