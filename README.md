# patternscale

Generic proxy-based pattern scaling: downscales regional scenario
trajectories (e.g. IAM emissions) to a grid using base-year grid patterns and
a gridded proxy trajectory (e.g. population, GDP), with optional corrections.

Method per variable, region R, grid cell g, base year t0:

$E(t,g) = E(t_0,g) \cdot \frac{E(t,R)}{E(t_0,R)} \cdot \left(\frac{P(t,g)}{P(t_0,g)} / \frac{P(t,R)}{P(t_0,R)}\right) \cdot N(t,R)$


where N normalizes regional grid sums to the scenario totals.

## Scope

Included: preprocessing of original data sources into the engine's input
stores (`preprocess/`: CEDS and EDGAR emission grids, COMPASS
population/GDP/urbanization projections, city typology, grid-to-country
region rasters — a clearly separate step that runs once per data release),
data contract validation, harmonization onto the target grid,
min-proxy floor, the scaling step, normalization, the new_dev correction
(newly developed cells set to regional mean intensity plus renormalization),
mask-restricted regional sums, result writers, and format-level adapters
(`adapters/`: SMIP country files, wide IAMC CSVs incl. REMIND processing,
emission/urbanization/region rasters; the required structure of adapter
outputs is documented in `adapters/__init__.py`). Excluded (application
code): directory layouts and filename conventions, multi-scenario
orchestration and all post-processing (shares, intensities, GHG conversion,
region re-aggregation, plots).

## Data contract

`downscale(scenario_df, ds, region_map, config, masks=..., mask_combinations=...)`
consumes (see `src/patternscale/contract.py` for the full definition):

1. `scenario_df`: long DataFrame with columns Variable, Region,
   Region_number, Year, Value — regional totals for every target variable and
   proxy, covering the base year and all target years.
2. `ds`: xarray Dataset on the target grid (dims y, x, time) with one data
   variable per target variable (base year required), per proxy (all years
   required) and the region-number raster.
3. `region_map`: DataFrame with Region and Region_number — the regions to
   downscale.

Inputs are validated on entry; violations raise `ContractError` listing all
problems.

## Minimal usage

```python
from patternscale import Config, downscale

config = Config.from_yaml("my_config.yaml")
config.meta.update({"model": "...", "scenario": "..."})

results = downscale(scenario_df, ds, region_map, config)
results.save_table("out/regional.csv")                  # wide (legacy layout)
results.save_table("out/regional_long.csv", "long")     # long (canonical)
results.save_grid("out/grid.zarr")                      # grids per variable
```

A complete application example (path conventions, harmonization, masks,
proxy aggregates) is `process_check_data/run_downscaling_patternscale.py` in
the IAM_downscaling_ScenarioMIP repository, with its configuration in
`configs/city_downscaling_smip.yaml` there.

## Tests

Synthetic invariants: mass conservation, base-year identity, analytic values
of the scaling step, correction behaviour, contract validation, config
round-trip.

```
python -m unittest discover -s tests -v
```

## Installation

Standard: `pip install -e .` (or `pixi add --pypi --editable <path>` in a
pixi project). If pip is unavailable (offline environment), point the
interpreter at `src/` via a `.pth` file in site-packages or via `PYTHONPATH`.

## Reproducibility notes

The implementation is kept bit-compatible with the original
`city_downscaling_main.py` (IAM_downscaling_ScenarioMIP repository) where
feasible: identical arithmetic expression order in the scaling step and
corrections, division (not multiplication by the reciprocal) for unit
conversions (`ProxySpec.intensity_divisor`), and the exact original
coordinate literals for the 0.1-degree grid.

## License

GNU Lesser General Public License v3 (LGPL-3.0), see `LICENSE`.

Note that the LGPL-3 text incorporates the terms of the GNU General Public
License v3 by reference. A copy of the GPL-3 text (conventionally `COPYING`,
from https://www.gnu.org/licenses/gpl-3.0.txt) should be added alongside
`LICENSE` before publishing or sharing.
