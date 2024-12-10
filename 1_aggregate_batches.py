import pandas as pd
from pathlib import Path 
from functools import reduce
from progiter import ProgIter
import warnings
import os

sp_cell_df_directory = Path('sp_cell_dfs/')
sp_cell_df_directory.mkdir(exist_ok=True)
batch_directory = Path('batches/')

def aggregate_by_cell(dataframes):
    """Aggregate column values in dataframes by H3 cells
    
    Args:
        dataframes: pd.DataFrame with index = H3 cell, identical columns
    """
    return reduce(lambda a, b: a.add(b, fill_value=0), dataframes)
    

def parse_batch_files(batch_directory):
    """Aggregate all species CSVs by resolution & report status
    
    Arguments:
        batch_directory: pathlib.Path containing CSVs to aggregate
    """
    batch_files = list(batch_directory.glob("*.csv"))
    
    # Files are named in the format:
    #f'{sp}_row{start_row}-{end_row}_status-{status}_resolution{resolution}.csv'
    file_info = [n.name.split("_") for n in batch_files]
    
    files = pd.DataFrame(file_info, columns=['SCIENTIFIC NAME', 'ROW RANGE', 'REPORT STATUS', 'RESOLUTION'])
    files['FILENAME'] = batch_files

    # Save CSVs of the aggregated data
    for (species, status, resolution), species_df in ProgIter(files.groupby(["SCIENTIFIC NAME", 'REPORT STATUS', 'RESOLUTION'])):
        print('\n')
        print(species)
        # Skip this file if we've already saved it
        species = species.replace(" ", "-")
        resolution = resolution[:-4] # remove .csv
        filename = sp_cell_df_directory.joinpath(f'{species}_{status}_{resolution}.csv')
        if filename.exists():
            print("Filename", filename.name, "exists. Continuing.")
            continue

        # Add subspecies dataframes' cell counts across batches
        all_dataframes = []
        for f in species_df.FILENAME:
            if os.path.getsize(f) == 0:
                warnings.warn(f'File "{f}" has size 0. Skipping.')
            else:
                all_dataframes.append(pd.read_csv(f, index_col=0))
        #all_dataframes = [pd.read_csv(f, index_col=0) for f in species_df.FILENAME]
        sp_cell_df = aggregate_by_cell(all_dataframes)

        # Save the CSV
        sp_cell_df.to_csv(filename)
        print("\nSaved", filename.name)
        quit()


parse_batch_files(batch_directory)