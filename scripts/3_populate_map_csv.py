import pandas as pd
from pathlib import Path

# Load eBird taxonomy
taxonomy = pd.read_csv("../resources/eBird_taxonomy_v2024.csv")

## Create a CSV of map URLs for the website
df = pd.DataFrame(columns=["common_name", "scientific_name", "resolution", "map_url"])
maps_dir = Path("../docs/maps")

# Sort maps taxonomically
maps_list = list(maps_dir.glob("*.html"))
species = {p.name.split('_')[0].replace('-', ' '): p for p in maps_list}
total_spp = len(species.keys())
ordering = taxonomy[['SCI_NAME', 'TAXON_ORDER']].set_index("SCI_NAME").to_dict()['TAXON_ORDER']
maps_list = sorted(maps_list, key=lambda x: ordering[x.name.split('_')[0].replace('-', ' ')])

# Create table of common name, species, resolution, and map URL
# This is used by the website
for idx, file in enumerate(maps_list):
    resolution = file.stem.split("_")[-1]
    species = file.stem.replace(f"_{resolution}", "")
    common_name = taxonomy[taxonomy['SCI_NAME'] == species.replace('-', ' ')].PRIMARY_COM_NAME.values[0]
    #map_url = Path(Path(file).parent.stem).joinpath(Path(file).name)
    map_url = 'https://subspeciesmapper.netlify.app/' + str(file.relative_to(maps_dir))
    print(map_url)
    df.loc[idx] = [common_name, species, resolution, map_url]
df.to_csv("../docs/data/map_data.csv", index=False)

print("Number of species mapped:", total_spp)
