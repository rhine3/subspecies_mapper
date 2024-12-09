import pandas as pd
from pathlib import Path 

sp_cell_df_directory = Path('sp_cell_dfs/')
sp_cell_df_directory.mkdir(exist_ok=True)

def parse_batch_files(batch_directory):
    batch_files = list(batch_directory.glob("*.csv"))
    
    # Files are named in the format:
    #f'{sp}_row{start_row}-{end_row}_status-{status}_resolution{resolution}.csv'
    file_info = [n.name.split("_") for n in batch_files]
    
    files = pd.DataFrame(file_info, columns=['SCIENTIFIC NAME', 'ROW RANGE', 'REPORT STATUS', 'RESOLUTION'])
    files['FILENAME'] = batch_files
    for (species, status, resolution), species_df in files.groupby(["SCIENTIFIC NAME", 'REPORT STATUS', 'RESOLUTION']):

        # Add subspecies dataframes' cell counts across batches
        all_dataframes = [pd.read_csv(f, index_col=0) for f in species_df.FILENAME] 
        sp_cell_df = reduce(lambda a, b: a.add(b, fill_value=0), all_dataframes)
        
        # Save a 
        species = species.replace(" ", "-")
        resolution = resolution[:-4] # remove .csv
        filename = sp_cell_df_directory.joinpath(f'{species}_{status}_{resolution}.csv')
        
        sp_cell_df.to_csv(filename)


parse_batch_files(batch_directory)