# Subspecies Plotter

This project seeks to create maps of the subspecies, identifiable forms, and intergrades for all species on eBird with *listable* subspecies (~1300).

### Scripts

Testing entire pipeline:

* For individual-species datasets: `get_geogrids.ipynb`
* For entire eBird dataset:`get_geogrids_fullebd.ipynb`

Scripts for individual steps in the pipeline, applied to entire eBird dataset (WIP):

* `0_make_batches_onerun.py` - count checklists/cell for species in EBD in batches
* `1_aggregate_batches.py` - aggregate (add) cell counts across batches
* `2_make_maps.py` - create the maps!


### Data used

```
eBird Basic Dataset. Version: EBD_relOct-2024. Cornell Lab of Ornithology, Ithaca, New York. Oct 2024.
```
