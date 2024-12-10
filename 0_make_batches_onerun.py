#import fireducks.pandas as pd
import pandas as pd
import pandas as pdr #For compatibility with fireducks version

from pathlib import Path
import h3
import matplotlib.pyplot as plt
import ast
pd.options.mode.copy_on_write = True
import json
from progiter import ProgIter
import os


# Load eBird taxonomy
taxonomy = pd.read_csv("eBird_taxonomy_v2024.csv")
taxonomy.head()

# Dictionary mapping species to their infraspecies, by category
with open("infraspecies_ebird.json") as f:
    spp_dict = json.load(f)
    
    
resolutions = [2, 3, 4]


def remove_duplicates(df):
    """Remove any species sightings duplicated across groups
    
    Args:
        df: a slice of the eBird dataset in a dataframe
    """
    # Remove duplicate checklists
    return df[df['GROUP IDENTIFIER'].isnull() | ~df[df['GROUP IDENTIFIER'].notnull()].duplicated(subset=["GROUP IDENTIFIER", "SCIENTIFIC NAME"],keep='first')]

def filter_by_status(
    df,
    status='approved' #otherwise: unflagged, flagged_approved, unconfirmed
):
    """Filter eBird records by flag and approval status
    
    Args:
        df: a slice of the eBird dataset in a dataframe
        confirmation_status:
            if 'approved': keep APPROVED = 1
            if 'unflagged': keep APPROVED = 1, REVIEWED = 0
            if 'flagged_approved': keep APPROVED = 1, REVIEWED = 1
            if 'unconfirmed': keep APPROVED = 0, REVIEWED = 1
            if anything else: *does not filter*
    """
    # Removed unconfirmed observations or reviewed observations, if desired
    if status == "unconfirmed":
        df = df[df["APPROVED"] == 0]
    elif status in ["approved", "unflagged", "flagged-approved"]:
        df = df[df["APPROVED"] == 1]
        if status == "unflagged":
            df = df[df["REVIEWED"] == 0]
        elif status == "flagged-approved":
            df = df[df["REVIEWED"] == 1]
        
    return df

def create_grid_cells(df, resolutions=[2,3,4]):
    """Convert latitudes and longitudes to H3 cells
    
    Adds H3 cells in every desired resolution to the source dataframe
    
    Args:
        df: a dataframe with columns LATITUDE and LONGITUDE
        resolutions: the H3 resolutions to use
    """
    def _latlongchecker(row, resolution):
        if type(row.LATITUDE) != float or type(row.LONGITUDE) != float:
            print("WHAT!!!")
            print(row.LATITUDE, type(row.LATITUDE))
            print(row.LONGITUDE, type(row.LONGITUDE))
        return h3.latlng_to_cell(row.LATITUDE, row.LONGITUDE, resolution)
    
    # Convert latitude and longitude to an H3 hexagon ID
    for resolution in resolutions:
        df[f'hex_id_{resolution}'] = df.apply(lambda row:  _latlongchecker(row, resolution), axis=1)

#     # This one is slightly slower
#     def _latlong_to_cells(row, resolutions=resolutions):
#         return [h3.latlng_to_cell(row.LATITUDE, row.LONGITUDE, resolution) for resolution in resolutions]
    
#     row_names = [f'hex_id_{resolution}' for resolution in resolutions]
#     df[row_names] = df.apply(_latlong_to_cells, axis='columns', result_type='expand')

    return df


def get_grid_cell_species_data(cell_df, sp, subspp, resolution):
    """Get # of checklists containing a species and each subspecies

    Args:
    - cell_df: pd.DataFrame, dataframe of all observations for a single grid cell (1 row per observation)
    - sp: str, scientific name of species
    - subspp: list of str, scientific names of subspecies for this species

    Returns:
    - cell_data: dict, with keys 'cell_id', species name, and subspecies names
    """
    # Total number of checklists containing the species
    num_checklists = cell_df["SAMPLING EVENT IDENTIFIER"].nunique()

    # Create a dict of # checklists containing sp for all cells
    try:
        cell_data = {'cell_id': cell_df[f"hex_id_{resolution}"].iloc[0]}
    except:
        print(cell_df)
    cell_data[sp] = num_checklists

    # Add number of checklists containing each subspecies
    for subsp in subspp:
        num_subsp = cell_df[cell_df["SUBSPECIES SCIENTIFIC NAME"] == subsp].shape[0]
        cell_data[subsp] = num_subsp

    return cell_data

def get_subspecies_df(sp, sp_df, subspp, resolution):
    """Make dataframe of species & subspecies data for every cell for a given species

    Args:
    - sp: str, scientific name of species
    - df: pd.DataFrame, dataframe of data for this species
    - subspp: list of str, scientific names of subspecies for this species
    - resolution: int, H3 resolution level
    """

    # Create a dict of # checklists containing sp for all cells
    cell_dicts = []
    for cell in sp_df[f"hex_id_{resolution}"].unique():
        cell_df = sp_df[sp_df[f"hex_id_{resolution}"] == cell]
        cell_data = get_grid_cell_species_data(cell_df, sp, subspp, resolution)
        cell_dicts.append(cell_data)

    sp_cell_df = pd.DataFrame(cell_dicts, index=range(len(cell_dicts)))
    sp_cell_df.set_index("cell_id", inplace=True)

    return sp_cell_df


def filter_species(df, keep_species):
    """Filter dataframe to only include species in the list
    """
    filtered = df[df["SCIENTIFIC NAME"].isin(keep_species)]
    kept_species = list(filtered["SCIENTIFIC NAME"].unique())
    return filtered, kept_species


def cast_types(df):
    """Get correct types for columns and remove any uncastable entires
    """
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce') 
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    return df.dropna(subset=['LATITUDE', 'LONGITUDE'])     # Drop rows without float latitudes


# The dictionary we've created of all species and their infraspecies, by category
spp_dict = json.load(open("infraspecies_ebird.json"))
spp_with_infras = [sp for sp, info in spp_dict.items() if len(info["infraspecies"].keys()) > 0]

# Only include the columns we're using
use_cols = [
        'SCIENTIFIC NAME', 'SUBSPECIES SCIENTIFIC NAME',
        'SAMPLING EVENT IDENTIFIER', 'GROUP IDENTIFIER',
        'LATITUDE', 'LONGITUDE', 
        'REVIEWED', 'APPROVED', 'OBSERVATION DATE']

# Process the whole eBird dataset
dataset_filepath = f"data/ebd_relOct-2024.txt"


# Where we will store each batch of data
batch_directory = Path('batches_onerun/')
batch_directory.mkdir(exist_ok=True)

statuses = ["unflagged", "flagged-approved"] 

# Read in CSV in batches
chunk_rows = 10000000

# Keep track of the progress made in each batch using a tracker CSV
Path("progress_trackers").mkdir(exist_ok=True)
tracker_filepath = f"progress_trackers/ebd_tracker_rowsperchunk-{chunk_rows}_onerun.csv"

# If already started processing, pick up where we left off 
if Path(tracker_filepath).exists():

    # Read the tracker in regular pandas (pdr)
    tracker = pdr.read_csv(tracker_filepath)
    tracker["spp_to_do"] = tracker["spp_to_do"].apply(ast.literal_eval) # Track which species are in this chunk
    tracker["spp_done"] = tracker["spp_done"].apply(ast.literal_eval) # Track what species are complete from this chunk
    start_idx = tracker.index[-1]
    spp_to_do = set(tracker.loc[start_idx].spp_to_do) - set(tracker.loc[start_idx].spp_done)
    
    # If we did all the spp in the last saved chunk, move to the next chunk
    if spp_to_do == set():
        prev_end_row = tracker.loc[start_idx].end_row
        prev_start_row = tracker.loc[start_idx].start_row
        if prev_end_row - prev_start_row < chunk_rows:
            print("No more data to process, total rows in dataset: ", prev_end_row)
            assert(False)
        skiprows = tracker.loc[start_idx].end_row # This is the row after the last one we processed
        start_idx = start_idx + 1
        spp_to_do = None
    else:
        skiprows = tracker.loc[start_idx].start_row

else:
    tracker = pdr.DataFrame(columns=["start_row", "end_row", "spp_to_do", "spp_done"])
    start_idx = 0
    spp_to_do = None
    skiprows = start_idx * chunk_rows


for idx, chunk in enumerate(pd.read_csv(
    dataset_filepath, chunksize=chunk_rows, header=0, 
    skiprows=range(1,skiprows),
    usecols=use_cols, 
    sep="\t", 
    keep_default_na=False  # Prevent treating empty strings as NaN
)):
    #print(f" ** Chunk memory usage - {sum(chunk.memory_usage()) * 0.000001} MB for {len(chunk.index)} Rows")
    
    # Check whether we are finishing or have finished all chunks 
    if chunk.shape[0] == 0:
        print("No more data to process, total rows in dataset: ", (start_idx + idx)*chunk_rows)
        break
    if chunk.shape[0] < chunk_rows:
        end_row = (start_idx+idx)*chunk_rows + chunk.shape[0]+1
        print(f"Last chunk, total rows in dataset:", end_row)
    else:
        end_row = (start_idx + idx)*chunk_rows+chunk_rows

    start_row = (start_idx + idx)*chunk_rows
    print("Start", start_row, "End", end_row)

    # Clean up the dataframe and subset to needed entries
    cleaned = cast_types(chunk)             # Prepare datatypes in the dataset
    cleaned = remove_duplicates(cleaned)    # Remove duplicate checklists
    cleaned, species_in_chunk = filter_species(
        cleaned, keep_species=spp_with_infras) # Remove spp with no reportable infraspecies

    # Get H3 cells for all observations
    cleaned = create_grid_cells(cleaned)       

    # If new row, determine which species there are to evaluate in the chunk
    if spp_to_do == None: # Add new row
        spp_to_do = species_in_chunk
        tracker.loc[start_idx+idx] = [(start_idx + idx)*chunk_rows, end_row, spp_to_do, []]

    # Do each species separately to easily pick back up where left off
    for sp in ProgIter(spp_to_do):
        for status in statuses:
            # Filter to set of species and observation statuses
            filtered = filter_by_status(cleaned, status)
            filtered, _ = filter_species(filtered, keep_species=[sp]) # Subset to the needed spp
            if filtered.shape[0] == 0: continue

            # Get list of countable subspecies from each ssp group
            subspp = []
            for ssp_type, ssp in spp_dict[sp]['infraspecies'].items(): subspp.extend(ssp.keys())

            # Save data on # obs of each subspp for each cell for each resolution
            for resolution in resolutions:
                filename = batch_directory.joinpath(f'{sp}_row{start_row}-{end_row}_status-{status}_resolution{resolution}.csv')
                batch_df = get_subspecies_df(sp, filtered, subspp, resolution)
                batch_df.to_csv(filename)

        # Update the tracker
        tracker.loc[0].spp_done += [sp]
        tracker.to_csv(tracker_filepath, index=False)

    spp_to_do = None
    
print("Done")