import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce
import os

#import _json
import json
pd.options.mode.copy_on_write = True
from progiter import ProgIter

import h3
import folium

from geojson import Feature, FeatureCollection
from shapely.geometry import Polygon, LineString, mapping
from shapely.ops import split

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import colorsys
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath import color_diff_matrix
#from colormath.color_diff import delta_e_cie2000 # deprecated and doesn't work anymore, reimplemented below

remake_maps = True
include_flagged_confirmed = True
resolutions = [3,2,4]

resources_dir = Path("../resources")
sp_cell_df_dir = resources_dir.joinpath("sp_cell_dfs")

# Load eBird taxonomy
taxonomy = pd.read_csv(resources_dir.joinpath("eBird_taxonomy_v2024.csv"))

# Load dict of infraspecies
with open(resources_dir.joinpath("infraspecies_ebird.json")) as f:
    spp_dict = json.load(f)
    
# Make 70 jobs, each of which does maps for 20 spp.
job_num = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
species_per_job = 20
start_idx = job_num*species_per_job
end_idx = (job_num+1)*species_per_job
    
# Extract which species scientific names have infraspecies associated with them
spp_with_infras = [sp for sp, info in spp_dict.items() if len(info["infraspecies"].keys()) > 0]
          
intergrade_parents = pd.read_csv(resources_dir.joinpath("ebird_2024_intergrade_parents.csv"))

def import_sp_cell_df(dataname):
    if not Path(dataname).exists():
         return None
    sp_cell_df = pd.read_csv(dataname, index_col=0)
    sp_cell_df.columns = sp_cell_df.columns.str.replace(species + ' ', "")
    return sp_cell_df

def aggregate_by_cell(dataframes):
    """Aggregate column values in dataframes by H3 cells
    
    Args:
        dataframes: pd.DataFrame with index = H3 cell, identical columns
    """
    return reduce(lambda a, b: a.add(b, fill_value=0), dataframes)


def get_infraspecies_relationships(sp, spp_dict=spp_dict, intergrade_parents=intergrade_parents):
    """Get a dictionary mapping infraspecies/forms to their "parents"
    
    Args:
        sp: scientific name of species
        spp_dict: dictionary of children of every species
        intergrade_parents: eBird-maintained CSV of intergrade parents
    """
    data = spp_dict[sp]['infraspecies']
    
    # Get a list of each type of infraspecies
    if "issf" in data.keys():
        issfs = [k.replace(sp+' ', '') for k in data["issf"].keys()] # Recognized ssp or ssp groups
    else:
        issfs = []
    if "form" in data.keys():
        forms = [k.replace(sp+' ', '') for k in data["form"].keys()] # Forms
    else:
        forms = []
    if "intergrade" in data.keys():
        intergrades = [k.replace(sp+' ', '') for k in data["intergrade"].keys()] # Intergrades (between ssp? forms?)
    else:
        intergrades = []

    intergrade_to_parents = dict()
    forms_to_parents = dict()
    top_level_intergrades = []
    top_level_forms = []

    # Find parents of the intergrades, if any are in the eBird taxonomy
    # Also determine which intergrades, if any, have no parents
    for intergrade in intergrades:
        # Come back to using the eBird CSV later
        # For Redpoll, problem is that one of the listed parents, "flammea Group", isn't in the main eBird Taxonomy, though "flammea" is
        # This could be a problem with other spp as well, so let's skip it for now
#         intergrade_searching = f"{sp} .{intergrade}"
#         parents = intergrade_parents[intergrade_parents.sci_name == intergrade_searching].parent_sci_name.values.tolist()
#         if len(parents) > 0:
#              intergrade_to_parents[intergrade] = parents
        
        parents = [i.strip() for i in intergrade.split('x')]
        # Check if all are true
        if all([p in issfs+forms for p in parents]):
            intergrade_to_parents[intergrade] = parents
        else:
            top_level_intergrades.append(intergrade)

    # Find parents of the forms, if any are in the eBird taxonomy
    # Also determine which forms, if any, have no parents
    for form in forms:
        # Split the form into its individual components
        form_parts = set(form.split("/"))
        
        parent_issfs = []
        for component in issfs:
            # Split the list component into subparts and check if all are in the form_parts
            component_parts = set(component.split("/"))
            if component_parts <= form_parts:  # Check if component_parts is a subset of form_parts
                parent_issfs.append(component)
        if len(parent_issfs):
            forms_to_parents[form] = parent_issfs
        else:
            top_level_forms.append(form)
    
    return issfs, forms, intergrades, intergrade_to_parents, forms_to_parents, top_level_intergrades, top_level_forms

def rgb_to_hex(rgb):
    """Convert an (R, G, B) tuple to a hex color (#RRGGBB)."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def hex_to_rgb(hex_color):
    """Convert hex color (#RRGGBB) to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def combine_rgb_colors(rgb_colors, fracs):
    """Combine a list of RGB colors proportionally."""
    if sum(fracs) == 0:
        return "#999999"
    else:
        combined_rgb = tuple(
            int(sum(frac * color[channel] for color, frac in zip(rgb_colors, fracs)))
            for channel in range(3)
        )
    return rgb_to_hex(combined_rgb)

def name_to_base_hue(name):
    """Generate a base hue from a name."""
    base_hue = hash(name) % 360
    return base_hue

def average_hues(hues):
    """Average a list of hues on the circular scale."""
    x = np.mean([np.cos(np.radians(h)) for h in hues])
    y = np.mean([np.sin(np.radians(h)) for h in hues])
    avg_hue = np.degrees(np.arctan2(y, x)) % 360
    return avg_hue


def delta_e_cie2000(color1, color2, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) of two colors.
    """
    def _get_lab_color1_vector(color):
        return np.array([color.lab_l, color.lab_a, color.lab_b])
    def _get_lab_color2_matrix(color):
        return np.array([(color.lab_l, color.lab_a, color.lab_b)])

    color1_vector = _get_lab_color1_vector(color1)
    color2_matrix = _get_lab_color2_matrix(color2)
    delta_e = color_diff_matrix.delta_e_cie2000(
        color1_vector, color2_matrix, Kl=Kl, Kc=Kc, Kh=Kh)[0]
    return delta_e

def rgb_to_lab(rgb):
    """Convert an RGB color (0-255) to LAB color space."""
    srgb = sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, is_upscaled=False)
    return convert_color(srgb, LabColor)

def hsl_to_rgb(hue, saturation, lightness):
    """Convert HSL values to RGB (0-255)."""
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    return (int(r * 255), int(g * 255), int(b * 255))


def is_color_too_similar(hue1, hue2, threshold=15):
    """
    Check if two hues are too similar, accounting for perceptual non-uniformity.
    Compare using the CIEDE2000 formula in the LAB color space.

    Higher threshold ==> colors need to be more different
    """

    # Convert hues to RGB colors using fixed saturation and lightness for comparison
    rgb1 = hsl_to_rgb(hue1, 0.8, 0.5)  # Vivid, medium lightness
    rgb2 = hsl_to_rgb(hue2, 0.8, 0.5)

    # Convert RGB to LAB for perceptual uniformity
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    # Calculate perceptual difference using CIEDE2000
    delta_e = delta_e_cie2000(lab1, lab2)
    return delta_e < threshold


def create_distribution_adjacency_matrix(data, subspecies_cols, cell_col='cell_id'):
    """
    Create an adjacency matrix based on subspecies distribution similarities.

    Parameters:
    - data: DataFrame with cells as rows and subspecies counts as columns.
    - subspecies_cols: List of column names corresponding to subspecies counts.
    - cell_col: Column name for cell identifiers (optional, for reference).

    Returns:
    - adjacency_matrix: A NumPy array where element [i, j] is the similarity between subspecies distributions.
    - subspecies_list: The order of subspecies corresponding to matrix rows/columns.
    """
    # Subset the subspecies columns
    subspecies_data = data[subspecies_cols]

    # Normalize each cell's counts to proportions
    subspecies_distribution = subspecies_data.div(subspecies_data.sum(axis=1), axis=0).fillna(0)

    # Compute cosine similarity between each pair of subspecies
    adjacency_matrix = cosine_similarity(subspecies_distribution.T)

    # Return the matrix and list of subspecies
    return adjacency_matrix, subspecies_cols


def hue_to_hex_vibrant(hue):
    saturation, lightness = 0.8, 0.5  # Vivid, medium colors
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    return rgb_to_hex((int(r * 255), int(g * 255), int(b * 255)))


def style_function(feature, subspp_colors):
    """Style a cell based on the proportion of subspecies."""
    properties = feature['properties']
    subspecies_values = dict()
    for subsp in subspp_colors.keys():
        val = properties.get(subsp, 0)
        if val > 0:
            subspecies_values[subsp] = val
    
    # Normalize the values to sum up to 1 for proportional allocation
    total = sum(subspecies_values.values())
    if total > 0:
        fracs = [value / total for value in subspecies_values.values()]
    else:
        fracs = [0 for _ in subspecies_values]
    
    # Get RGB colors for each subspecies
    hex_colors = [subspp_colors[subsp] for subsp in subspecies_values]
    rgb_colors = [hex_to_rgb(color) for color in hex_colors]
    
    # Combine colors based on the proportional fractions
    cell_color = combine_rgb_colors(rgb_colors, fracs)
    
    return {
        'fillColor': cell_color,  # Cell color
        'color': cell_color,  # Border color
        'weight': 1,  # Border weight
        'fillOpacity': 0.6,  # Cell fill transparency
    }

def calculate_overlap_intensity(overlap_matrix):
    """
    Calculate the overlap intensity for each subspecies.
    Overlap intensity is defined as the proportion of cells that overlap with others.

    Parameters:
    - overlap_matrix: A square matrix where overlap_matrix[i][j] represents the overlap between subspecies[i] and subspecies[j].

    Returns:
    - A list of overlap intensities for each subspecies.
    """
    total_overlap = np.sum(overlap_matrix, axis=1)  # Total overlap for each subspecies
    max_overlap = np.sum(overlap_matrix, axis=1) - np.diagonal(overlap_matrix)  # Exclude self-overlap
    return total_overlap / max_overlap


def generate_priority_hues(subspecies, overlap_matrix):
    """
    Assign perceptually distinct hues to subspecies, prioritizing those with higher overlap intensity.

    Parameters:
    - subspecies: List of subspecies names.
    - overlap_matrix: A square matrix where overlap_matrix[i][j] represents the overlap between subspecies[i] and subspecies[j].

    Returns:
    - A dictionary mapping each subspecies to a distinct hue (in degrees, 0-360).
    """
    n_subspecies = len(subspecies)

    # Step 1: Generate perceptually distinct hues (0-360 degrees)
    hues = np.linspace(0, 360, n_subspecies, endpoint=False)

    # Step 2: Calculate overlap intensity
    overlap_intensity = calculate_overlap_intensity(overlap_matrix)

    # Step 3: Assign hues based on overlap intensity and relationships
    G = nx.Graph()
    for i, sp1 in enumerate(subspecies):
        for j, sp2 in enumerate(subspecies):
            if overlap_matrix[i][j] > 0.1:  # Add edge if overlap is above threshold
                G.add_edge(sp1, sp2, weight=overlap_matrix[i][j])

    # Sort subspecies by overlap intensity
    subspecies_sorted = sorted(zip(subspecies, overlap_intensity), key=lambda x: x[1], reverse=True)
    zero = subspecies_sorted[0]
    one = subspecies_sorted[1]
    
    # For dark-eyed junco, overrule the ordering for fun
    if 'aikeni' == zero[0] and 'dorsalis' == one[0]:
        subspecies_sorted[0] = one
        subspecies_sorted[1] = zero
    
    # Assign hues sequentially to prioritize highly overlapping subspecies
    assigned_hues = {}
    used_hues = set()
    for subsp, _ in subspecies_sorted:
        #if subsp = 
        # Find the most distinct unused hue
        best_hue = None
        max_dist = -1
        for i, hue in enumerate(hues):
            if i in used_hues:
                continue
            # Check perceptual distance to already assigned hues
            if assigned_hues:
                dist = np.min(
                    [min(abs(hue - assigned_hues[sp]), 360 - abs(hue - assigned_hues[sp])) for sp in assigned_hues]
                )
            else:
                dist = float("inf")
            
            if dist > max_dist:
                max_dist = dist
                best_hue = i

        # Assign the best hue
        assigned_hues[subsp] = hues[best_hue]
        used_hues.add(best_hue)
    
    return assigned_hues


def get_bounds(geojson_result):
    """
    Calculate the bounding box of all features in the GeoJSON.

    Args:
    - geojson_result: GeoJSON string with features.

    Returns:
    - Bounds as [[southwest_lat, southwest_lon], [northeast_lat, northeast_lon]].
    """
    import json
    #geojson_data = json.loads(geojson_result)
    geojson_data = geojson_result
    all_coords = []

    for feature in geojson_data['features']:
        # Extract all coordinates from the polygon or multipolygon
        coords = feature['geometry']['coordinates']
        if feature['geometry']['type'] == "Polygon":
            all_coords.extend(coords[0])  # Add outer ring of the polygon
        elif feature['geometry']['type'] == "MultiPolygon":
            for poly in coords:
                all_coords.extend(poly[0])  # Add outer ring of each polygon

    # Extract longitudes (x) and latitudes (y) correctly
    lons, lats = zip(*all_coords)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def split_polygon_at_line(polygon, line):
    """
    Splits a GeoJSON polygon at a given GeoJSON line.
    
    Parameters:
        polygon (dict): shapely representation of the polygon.
        line (dict): shapely representation of the line.
    
    Returns:
        list: A list of GeoJSON polygons after splitting.
    """
    
    # Perform the split
    split_result = split(polygon, line)
    
    # Convert Shapely polygons back to GeoJSON
    split_geojsons = [mapping(geom) for geom in split_result.geoms]
    
    return split_geojsons


def split_at_dateline(geojson):
    """
    Adjusts polygons in a GeoJSON that cross the International Date Line (180° longitude)
    by adding 360 to longitudes less than 0, but only for polygons within 10 degrees of the dateline.
    
    Parameters:
    - geojson: A GeoJSON-like dictionary of geometries.
    
    Returns:
    - A new GeoJSON with adjusted geometries.
    """
    adjusted_features = []

    for feature in geojson['features']:
        geom = feature['geometry']
        coords = geom['coordinates']

#         if geom['type'] == 'Polygon':
        assert geom['type'] == 'Polygon'
            # Check if the polygon is near the dateline
        if any(abs(lon) > 165 for ring in coords for lon, _ in ring):
#       # Adjust longitudes for all rings in the polygon
#             new_coords = [
#                 [(lon + 360 if (lon < 0 and lat < 0) else lon, lat) for lon, lat in ring]
#                 for ring in coords
#             ]

            new_coords = []
            for ring in coords:
                for lon, lat in ring:
                    if (lon < 0 and (lat < 0)): # For the area of Fiji
                        new_coords.append((lon+360, lat))
                    elif (lon > 0 and (lat > 50)): # For Aleutians & far eastern Russia
                        new_coords.append((lon-360, lat))
                    else:
                        new_coords.append((lon, lat))
            geom['coordinates'] = [new_coords]

#         elif geom['type'] == 'MultiPolygon':
#             # Check each polygon in the MultiPolygon
#             new_coords = []
#             for polygon in coords:
#                 if any(abs(lon) > 170 for ring in polygon for lon, _ in ring):
#                     # Adjust longitudes for all rings in the polygon
#                     adjusted_polygon = [
#                         [(lon + 360 if (lon < 0 and lat < 0) else lon, lat) for lon, lat in ring]
#                         for ring in polygon
#                     ]
#                     new_coords.append(adjusted_polygon)
#                 else:
#                     # Keep polygon unchanged
#                     new_coords.append(polygon)
#             geom['coordinates'] = new_coords

        # Add the updated feature to the new collection
        adjusted_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": feature['properties']
        })

    return {
        "type": "FeatureCollection",
        "features": adjusted_features
    }

def get_range_size(sp_cell_df, ssp):
    """Calculate the range size of a subspecies."""
    return (sp_cell_df[ssp] > 0).astype(int).sum()

def get_color_mapping(sp_cell_df):
    def _average_and_adjust(parents, infraspecies):
        avg = average_hues([ssp_hues[parent] for parent in parents])
        start = avg
        amt_change = 0
        # Change any that look too similar to others
        while any(is_color_too_similar(avg, h2) for h2 in ssp_hues.values()):
            adj = 5
            avg -= adj % 360
            amt_change += adj
            if amt_change > 360: # Couldn't find a different enough color. looped around.
                print("Couldn't find more different color for", infraspecies)
                break
        return avg
    # Get the relationships between the infraspecies (ISSFs, forms, intergrades)
    issfs, forms, intergrades, inter_to_p, form_to_p, top_inter, top_form = get_infraspecies_relationships(species, spp_dict)

    # Get colors for the top-level infraspecies based on their geographic overlap
    top_level_infras = [*issfs, *top_inter, *top_form]
    sp_cell_df.columns = sp_cell_df.columns.str.replace(species + ' ', "")
    overlap_matrix, top_level_infras = create_distribution_adjacency_matrix(sp_cell_df, top_level_infras)
    ssp_hues = generate_priority_hues(top_level_infras, overlap_matrix)

    # Get colors for the lower-level infraspecies by averaging their "parent" infraspecies
    for intergrade, parents in inter_to_p.items():
        ssp_hues[intergrade] = _average_and_adjust(parents, intergrade)
        
    for form, parents in form_to_p.items():
        ssp_hues[form] = _average_and_adjust(parents, form)
#         ssp_hues[form] = average_hues([ssp_hues[parent] for parent in parents])
    
    

    # Convert hues to vibrant colors
    ssp_colors = {ssp: hue_to_hex_vibrant(hue) for ssp, hue in ssp_hues.items()}

    # Organize subspecies into categories and calculate their range sizes (from largest to smallest)
    sorted_issfs = sorted(issfs, key=lambda ssp: get_range_size(sp_cell_df, ssp), reverse=True)
    sorted_forms = sorted(forms, key=lambda ssp: get_range_size(sp_cell_df, ssp), reverse=True)
    sorted_intergrades = sorted(intergrades, key=lambda ssp: get_range_size(sp_cell_df, ssp), reverse=True)

    return ssp_colors, sorted_issfs, sorted_forms, sorted_intergrades

def get_ssp_common_name(species, subsp, taxonomy=taxonomy):
    # Look up the PRIMARY_COM_NAME for the subspecies
    subsp_name = taxonomy[taxonomy['SCI_NAME'] == f"{species} {subsp}"].PRIMARY_COM_NAME.values[0]
    # Get the part in the parentheses
    subsp_name = subsp_name.split('(')[-1].split(')')[0] if '(' in subsp_name else subsp
    return subsp_name


def reduce_precision(geojson_data, precision):
    def round_coords(coords):
        return [[round(coord[0], precision), round(coord[1], precision)] for coord in coords]
    
    def process_feature(feature):
        if feature['geometry']['type'] == 'Polygon':
            feature['geometry']['coordinates'] = [round_coords(coords) for coords in feature['geometry']['coordinates']]
        elif feature['geometry']['type'] == 'MultiPolygon':
            feature['geometry']['coordinates'] = [
                [round_coords(coords) for coords in polygon]
                for polygon in feature['geometry']['coordinates']
            ]
        return feature

    geojson_data['features'] = [process_feature(feature) for feature in geojson_data['features']]
    return geojson_data

def choropleth_map(sp_cell_df, common_name, subspp_colors, sorted_issfs, sorted_forms, sorted_intergrades, taxonomy=taxonomy):
    """Creates a choropleth map given species data."""
    
    f = folium.Figure()
    map = folium.Map(location=[47, -122], zoom_start=5, 
                     tiles="cartodbpositron", 
                     #tiles=folium.TileLayer("cartodbpositron", no_wrap=True),
                     control_scale=True, world_copy_jump=True)
    f.add_child(map)

    sp = sp_cell_df.columns[0]
    subspp = sp_cell_df.columns[1:]
    
    # Create polygon features with tooltips
    list_features = []
    for _, row in sp_cell_df.iterrows():
        # Calculate the relative percentage of sightings of each subspecies for each cell
        percentages = (row[subspp] / sum(row[subspp])) * 100
        percentages = round(percentages, 0)
        percentages_dict = percentages.to_dict()
        
        # Calculate total reports across all subspecies for each cell
        total_reports = row[subspp].sum()

        # Precompute tooltip text showing only subspecies with nonzero percentages
        percentages_dict_ordered = pd.DataFrame(percentages_dict, index=['pct']).T.query('pct > 0')['pct'].sort_values(ascending=False).to_dict()
        
        # Start tooltip container
        tt1_style = '''.tt1 {min-width: 100px; max-width: 300px; overflow: auto;}'''
        tooltip_text = '<div class="tt1">'

        # Add header and create a flex container for total reports and percentages
        tt2_style = '''.tt2 {display: flex; justify-content: space-between; margin-top: 5px;}'''
        tt3_style = '''.tt3 {margin-right: 10px;}'''
        tt4_style = '''.tt4 {text-align: right; background: ;}'''
        total_reports = row[subspp].sum()
        # Add percentages to the right-aligned div
        if total_reports > 0:
            tooltip_text += f'''<div class="tt2"><div class="tt3"><strong>Reported subspecies</strong><br>Total reports: {total_reports:.0f}</div><div class="tt4">'''
            for subsp, percent in percentages_dict_ordered.items():
                if percent > 0:
                    subsp_common_name = get_ssp_common_name(sp, subsp)
                    tooltip_text += f"<div>{subsp_common_name}: {percent:.0f}%</div>" if percent > 0.5 else ""
        else:
            tooltip_text += f'''<div class="tt2"><div class="tt3"><strong>Reported subspecies</strong><br></div><div class="tt4">'''
            tooltip_text += '''</div>None</div>'''

        # Close all divs
        tooltip_text += '''</div></div></div>'''

        # Add the tooltip to the feature's properties
        percentages_dict_ordered["tooltip"] = tooltip_text

        # Convert the H3 cell into a geometry
        geometry_for_row = h3.cells_to_geo(cells=[row.name])

        # Add a GeoJSON Feature to the list of features
        feature = Feature(
            geometry=geometry_for_row,
            id=row.name,
            properties=percentages_dict_ordered
        )
        list_features.append(feature)

    # Create a GeoJSON FeatureCollection with the list of features
    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)

    # Deal with geometries that cross the International Date Line
    geojson_result = split_at_dateline(json.loads(geojson_result))
    #geojson_result = adjust_longitudes(json.loads(geojson_result))

    # Reduce precision to 3 digits (111 meters) 
    geojson_result = reduce_precision(geojson_result, precision=3)
    
    # Add GeoJSON layer to the map
    folium.GeoJson(
        geojson_result,
        style_function=lambda feature: style_function(feature, subspp_colors),
        name=f'{sp} Subspecies Map'
    ).add_to(map)
    
    # Add a tooltip to the GeoJSON layer
    folium.GeoJson(
        geojson_result,
        style_function=lambda feature: {
            'weight': 0, 
            'color': 'transparent',  
            'fillOpacity': 0.6 
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip"],
            aliases=[None],  # No additional label prepended
            localize=True,
            sticky=True,
            labels=False,  # Disable default labels
            html=True  # Enable HTML formatting in the tooltip
        )
    ).add_to(map)


    # Add legend with subheaders
    legend_html = f"""
    <div style="position: fixed; top: 10px; right: 10px; max-width: 200px; height: auto; z-index: 9999; background-color: white; box-shadow: 0 0 5px rgba(0, 0, 0, 0.2); border: 1px solid lightgray; border-radius: 5px; padding: 10px; font-size: 10px;">
        <div style="font-size: 12px;"><strong>{common_name}</strong></div>
    """
    
    # Add subheaders and sorted species
    issf_colors = {k: subspp_colors[k] for k in sorted_issfs}
    form_colors = {k: subspp_colors[k] for k in sorted_forms}
    intergrade_colors = {k: subspp_colors[k] for k in sorted_intergrades}
    categories = [("Subspecies", issf_colors), ("Forms", form_colors), ("Intergrades", intergrade_colors)]
    # Remove any categories where the dictionary has no keys
    categories = [(category, ssp_in_category) for category, ssp_in_category in categories if ssp_in_category]

    # Create species and colors part of legend
    for category, sorted_subspecies in categories:
        legend_html += f'<div style="margin-top:10px;"><b>{category}</b></div>'
        for subsp in sorted_subspecies:
            subsp_common_name = get_ssp_common_name(sp, subsp)
            if subsp_common_name == subsp.strip('[]').strip('()'): # Only display one name if no unique common name
                display_name = subsp.strip('[]').strip('()')
            else:
                if subsp[0] == '(' or subsp[0] == '[':
                    display_name = f"{subsp_common_name} {subsp}"
                else:
                    display_name = f"{subsp_common_name} ({subsp})"

            display_name = '/<wbr>'.join(display_name.split('/')) # Add available wordbreaks for long slash names
            color = subspp_colors.get(subsp, "#ccc")  # Default color if no color is found

            # Get link to the subspecies's eBird species account
            subsp_code = taxonomy[taxonomy['SCI_NAME'] == f"{sp} {subsp}"].SPECIES_CODE.values[0]
            subsp_link = f"https://ebird.org/species/{subsp_code}"
            
            # Create a string for displaying subspp name
            subsp_display = f"""
            <div style="display: inline-block; max-width: 150px; white-space: normal; overflow-wrap: break-word;">
                <a href="{subsp_link}" target="_blank">{display_name}</a>
            </div>
            """
            # Combine formatted subspp name and block of color
            legend_html += f"""
            <div style="margin-top: 5px;">
                <span style="display: inline-block; width: 20px; height: 10px; margin-right: 5px; background-color: {color};"></span>
                {subsp_display}
            </div>
            """
    # Add a block of gray indicating no subspp reported
    legend_html += f"""
    <div style="margin-top: 10px;">
        <span style="display: inline-block; width: 20px; height: 10px; margin-right: 5px; background-color: #bbb;"></span>
         <i>No subspecies reported</i>
    </div>
    """
    
    legend_html += "</div>"
    
    legend_element = folium.Element(legend_html)
    map.get_root().html.add_child(legend_element)
    map.get_root().html.add_child(folium.Element(f"""
        <style>
            {tt1_style}
            {tt2_style}
            {tt3_style}
            {tt4_style}
        </style>"""))

    # Calculate bounds and adjust the map's view
    bounds = get_bounds(geojson_result)
    map.fit_bounds(bounds)

    string_so_far = map.get_root().render()
    return string_so_far, map
    
remake_maps=True
for idx, species in ProgIter(enumerate(spp_with_infras[start_idx:end_idx])):
# species below are all good examples
#for idx, species in ProgIter(enumerate(spp_with_infras)):
#for idx, species in ProgIter(enumerate(['Branta bernicla'])): #Brant
#for idx, species in ProgIter(enumerate(['Aplonis tabuensis'])):
#for idx, species in ProgIter(enumerate(['Larus glaucoides'])): #GLGU
#for idx, species in ProgIter(enumerate(['Chroicocephalus novaehollandiae'])): #Silver Gull
#for idx, species in ProgIter(enumerate(['Columba livia'])): #ROPI
#for idx, species in ProgIter(enumerate(['Zonotrichia leucophrys'])): #WCSP good example of intergrade color being almost identical to another color
#for idx, species in ProgIter(enumerate(['Junco hyemalis'])): #DEJU
#for idx, species in ProgIter(enumerate(['Saxicola maurus'])): # Siberian stonechat
#for idx, species in ProgIter(enumerate(['Buteo jamaicensis'])):
#for idx, species in ProgIter(enumerate(['Columba vitiensis'])):
#for idx, species in ProgIter(enumerate(['Leucosticte tephrocotis'])): #GCRF, good example for Aleutians
#for idx, species in ProgIter(enumerate(['Coccothraustes vespertinus'])): #EVGR, good example for labels
#for idx, species in ProgIter(enumerate(['Acanthis flammea'])): #Redpoll, good example of weird intergrade situation
    subspp_colors = None
    print("\nMapping", species)
    common_name = taxonomy[taxonomy['SCI_NAME'] == species].PRIMARY_COM_NAME.values[0]
    print(common_name)
    for resolution in resolutions:
        # Skip this species & resolution if needed
        map_filename = f"../docs/maps/{species.replace(' ', '-')}_{resolution}.html"
        if Path(map_filename).exists() and not remake_maps:
            print("Map already exists for", species, "at resolution", resolution)
            continue
        
        # Load datasets as needed
        if include_flagged_confirmed:
            # Aggregate both flagged and unflagged data
            df1 = import_sp_cell_df(sp_cell_df_dir.joinpath(f"{species.replace(' ', '-')}_status-unflagged_resolution{resolution}.csv"))
            df2 = import_sp_cell_df(sp_cell_df_dir.joinpath(f"{species.replace(' ', '-')}_status-flagged-approved_resolution{resolution}.csv"))
            if df1 is None:
                if df2 is None:
                    print("No data for", species, "at resolution", resolution)
                    continue
                else:
                    sp_cell_df = df2
            else:
                if df2 is None:
                    sp_cell_df = df1
                else:
                    sp_cell_df = aggregate_by_cell([df1, df2])    
        else:
            # Use only unflagged data
            dataname = sp_cell_dfs_dir.joinpath(f"{species.replace(' ', '-')}_status-unflagged_resolution{resolution}.csv")
            sp_cell_df = import_sp_cell_df(dataname)
        
        if sp_cell_df[sp_cell_df.columns[1:]].sum().sum() == 0:
            print("No subspecies reported to eBird. Continuing")
            continue

        subspecies = sp_cell_df.columns[1:]
        if resolution == 3 or subspp_colors == None:
            subspp_colors, sorted_issfs, sorted_forms, sorted_intergrades = get_color_mapping(sp_cell_df)
            
        map_string, m = choropleth_map(sp_cell_df, common_name, subspp_colors, sorted_issfs, sorted_forms, sorted_intergrades)
        map_filename = f"../docs/maps/{species.replace(' ', '-')}_{resolution}.html"
        m.save(map_filename)
        with open(map_filename, 'w') as f:
            f.write(map_string)