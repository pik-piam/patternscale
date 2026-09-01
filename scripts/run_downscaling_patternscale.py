#!/usr/bin/env python
"""City downscaling via the patternscale package (standalone repository).

Runs one scenario (default: set below) or, on a cluster, one scenario per
SLURM array task:

    python scripts/run_downscaling_patternscale.py                    # single run
    python scripts/run_downscaling_patternscale.py --task-id 3        # SCENARIOS[3]
    python scripts/run_downscaling_patternscale.py --scenario "SSP2 - Low Emissions"

Expects the preprocessed inputs under <repo>/data (see the path resolution
below); results are written to <repo>/results/<scenario>_<timestamp>.

Outputs:
- ``emi_grid_<timestamp>.zarr``          : downscaled emission grids
- ``downscaled_regional_<...>.csv``      : regional sums, mask sums (urban,
                                           peri-urban, typologies) and
                                           normalization factors (wide + long)
- ``proxy_aggregates_<...>.csv``         : urban/typology population and GDP
                                           aggregates
- ``run_config.yaml``                    : the typed run configuration

Post-processing (sector totals, shares, intensities, GHG, region
re-aggregation, plots) is not part of this script.
"""

import argparse
from datetime import datetime
from pathlib import Path
import logging
import subprocess
import warnings

import pandas as pd
import xarray as xr
from dask.array.core import PerformanceWarning
from pandas.errors import PerformanceWarning as PandasPerformanceWarning

import patternscale as ps

import project_io  # data paths and filename conventions (same directory)

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=PandasPerformanceWarning)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SCENARIOS = [
    "SSP1 - Very Low Emissions",
    "SSP2 - Low Emissions",
    "SSP2 - Medium Emissions",
    "SSP2 - Low Overshoot_a",
    "SSP3 - High Emissions",
    "SSP5 - Medium-Low Emissions_a",
    "SSP2 - Medium-Low Emissions",
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-id", type=int, default=None,
                    help="index into SCENARIOS (SLURM array mode)")
parser.add_argument("--scenario", type=str, default="SSP3 - High Emissions",
                    help="scenario name (ignored if --task-id is given)")
parser.add_argument("--set-name", type=str, default="",
                    help="suffix for the results folder, e.g. '_newDev_3'")
args = parser.parse_args()

scenario_name = SCENARIOS[args.task_id] if args.task_id is not None else args.scenario
set_name = args.set_name

smip_version = "v1-1-1"
# exact on-disk casing (case-sensitive filesystems, e.g. the cluster)
template_regions = "SMIP_countries_v1-1-1"
ssp = scenario_name.split(" - ")[0]
resolution_km = 12

# --- paths: everything relative to this repository ---------------------------
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
data_dir = repo_root / "data"

def _first_existing(candidates: list[Path], probe: str, kind: str) -> Path:
    for c in candidates:
        if list(c.glob(probe)):
            return c
    raise FileNotFoundError(
        f"No {kind} data found (looked for '{probe}' in: "
        f"{', '.join(str(c) for c in candidates)}). "
        f"Adjust the path candidates in {__file__}."
    )

processed_dir = _first_existing(
    [data_dir / "processed", data_dir], "Emissions_all_*_cf_*.zarr", "processed")
mappings_dir = _first_existing(
    [data_dir / "mappings", data_dir / "input" / "mappings", data_dir, processed_dir],
    "numbermapping_*.csv", "mapping")

now = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = repo_root / "results" / f"{scenario_name}_{now}{set_name}"

# scientific settings live in one place: the YAML config
config_path = repo_root / "configs" / "city_downscaling_smip.yaml"
if not config_path.is_file():
    config_path = script_dir / "city_downscaling_smip.yaml"
config = ps.Config.from_yaml(config_path)
years_downscaling = config.downscaling.years
target_variables = {v.short: v.name for v in config.downscaling.variables}

# run-specific settings
config.output.dir = str(output_dir)
config.logging.file = str(output_dir / f"{scenario_name}_{now}.log")

# provenance: identify the exact code state of this run
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
    ).strip()
except Exception:
    git_commit = "unknown"

config.meta.update({
    "scenario": scenario_name,
    "model": "auto-detect",  # set after reading the scenario file
    "SSP": ssp,
    "template_regions": template_regions,
    "data_format": template_regions,
    "smip_version": smip_version,
    "population_source": "Compass",
    "gdp_source": "Compass",
    "emissions_source": "CEDS",
    "urbanization_source": "Compass_pop",
    "typology_source": "Zhixin",
    "resolution_km": resolution_km,
    "timestamp": now,
    "patternscale_version": ps.__version__,
    "git_commit": git_commit,
})

output_dir.mkdir(parents=True, exist_ok=True)
ps.setup_logging(config.logging)
logging.info(f"patternscale run: {scenario_name}")
logging.info(f"data: processed={processed_dir}, mappings={mappings_dir}")

# ----------------------------------------------------------------------------
# 1. Data import
# ----------------------------------------------------------------------------
pop = project_io.load_population(processed_dir, "Compass", ssp, resolution_km)
gdp = project_io.load_gdp(processed_dir, "Compass", ssp, resolution_km)
emi = project_io.load_emissions(processed_dir, "CEDS", resolution_km, list(target_variables.values()))
urb = project_io.load_urbanization(processed_dir, "Compass_pop", ssp, years_downscaling)
typ = project_io.load_typology(processed_dir, "Zhixin", resolution_km)
gtr, region_map = project_io.load_region_raster(mappings_dir, template_regions, resolution_km)

df_IAM_all, df_IAM, model = project_io.load_smip_scenario(
    processed_dir, scenario_name, smip_version, target_variables)
config.meta["model"] = model
logging.info(f"Detected model: {model}")

config.to_yaml(output_dir / "run_config.yaml")

# ----------------------------------------------------------------------------
# 2. Cleaning and harmonization
# ----------------------------------------------------------------------------
grid = ps.Grid(config.downscaling.grid.resolution_deg)
gtc = ps.harmonize_dataset(gtr, grid, years=None, check_sum=False)
emi = ps.harmonize_dataset(emi, grid, years=None)
gdp = ps.harmonize_dataset(gdp, grid, years=years_downscaling)
pop = ps.harmonize_dataset(pop, grid, years=years_downscaling)
urb = ps.harmonize_dataset(urb, grid, years=years_downscaling)
typ = ps.harmonize_dataset(typ, grid, years=None)

ds = xr.merge([gtc, emi, gdp, pop, urb, typ])
ds = ds.chunk({"x": -1, "y": -1})

# scenario data: region numbers, year filter, N2O kt -> Mt
df_IAM = ps.smip.clean_scenario(df_IAM, region_map, years_downscaling)

# drop regions that do not report all needed variables (legacy check used all
# variables present in the filtered scenario frame, incl. auxiliary ones)
df_IAM, region_map, dropped = ps.restrict_to_complete_regions(
    df_IAM, region_map, config.downscaling,
    variables=df_IAM["Variable"].unique().tolist(),
)
if dropped:
    logging.info(f"Dropped regions without complete data: {dropped}")

# ----------------------------------------------------------------------------
# 3. Downscaling
# ----------------------------------------------------------------------------
masks = {
    "Urban": ds["mask_urb"],
    "Peri-Urban": ds["mask_peri"],
    "Type 1": ds["Type 1"],
    "Type 2": ds["Type 2"],
    "Type 3": ds["Type 3"],
    "Type 4": ds["Type 4"],
}
mask_combinations = {"Urban and Peri-Urban": ["Urban", "Peri-Urban"]}

results = ps.downscale(df_IAM, ds, region_map, config, masks=masks, mask_combinations=mask_combinations)

# ----------------------------------------------------------------------------
# 4. Proxy aggregates (urban/typology population and GDP)
# ----------------------------------------------------------------------------
# the pipeline applies the min-proxy floor internally to its own copy; apply
# it here as well so the proxy aggregates use the same floored fields
# (idempotent)
ds_prox = ds.copy()
for p in config.downscaling.proxies.values():
    if config.downscaling.corrections.min_proxy and p.min_value is not None:
        ds_prox[p.grid_var] = ps.apply_min_proxy(ds_prox[p.grid_var], p.min_value, grid)

logging.info("Calculating urban/typology population and GDP aggregates")
proxy_specs = [("Population", "Population", 1e6), ("GDP|PPP", "GDP", 1e9)]
proxy_frames = []
for scenario_var, grid_var, scale in proxy_specs:
    sums = {
        label: ps.regional_mask_sums(ds_prox[grid_var], ds_prox["region_number"], region_map, m, scale)
        for label, m in masks.items()
    }
    upu = pd.merge(sums["Urban"], sums["Peri-Urban"], on=["region", "year"])
    upu["value"] = upu["value_x"] + upu["value_y"]
    upu.drop(columns=["value_x", "value_y"], inplace=True)

    for label, df in sums.items():
        df["variable"] = f"{scenario_var}|{label}"
    upu["variable"] = f"{scenario_var}|Urban and Peri-Urban"

    proxy_frames.extend([sums["Urban"], sums["Peri-Urban"], upu,
                         sums["Type 1"], sums["Type 2"], sums["Type 3"], sums["Type 4"]])

proxy_agg = pd.concat(proxy_frames, axis=0, ignore_index=True)
proxy_agg["scenario"] = scenario_name
proxy_agg["model"] = model
proxy_agg = proxy_agg[proxy_agg["year"].isin(years_downscaling)]

# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------
if config.output.save_tables:
    # wide for legacy comparison, long as canonical (type-stable) format
    results.save_table(output_dir / f"downscaled_regional_{scenario_name}_{now}.csv")
    results.save_table(output_dir / f"downscaled_regional_long_{scenario_name}_{now}.csv", format="long")
    proxy_agg.to_csv(output_dir / f"proxy_aggregates_{scenario_name}_{now}.csv", index=False)
if config.output.save_grid:
    results.save_grid(output_dir / f"emi_grid_{now}.zarr")

logging.info(f"Done. Results in {output_dir}")
logging.shutdown()
