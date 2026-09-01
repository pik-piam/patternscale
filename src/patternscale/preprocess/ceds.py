"""Preprocessing of CEDS CMIP7 emission fluxes.

Input: monthly sectoral emission fluxes [kg m-2 s-1] on a 0.1-degree grid
(``<gas>-em-anthro_input4MIPs_..._gr_*.nc``, one directory per gas).
Output: annual sectoral emission grids [t/yr] with IAMC-style variable names
``Emissions|<gas>|<sector>``, dims (y desc, x, time), zeros masked to NaN.

Faithful port of ``process_emissions_CEDS.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# CEDS sector index -> CEDS sector name
SECTOR_NAMES = {
    0: "Agriculture",
    1: "Energy",
    2: "Industrial",
    3: "Transportation",
    4: "Residential, Commercial, Other",
    5: "Solvents production and application",
    6: "Waste",
    7: "International Shipping",
}

# CEDS sector name -> IAMC-style sector path (collisions are summed,
# e.g. Industrial + Solvents -> Industry)
SECTOR_MAPPING = {
    "International Shipping": "Transportation|Freight|International Shipping|Demand",
    "Energy": "Energy|Supply",
    "Industrial": "Industry",
    "Residential, Commercial, Other": "Energy|Demand|Residential and Commercial",
    "Transportation": "Energy|Demand|Transportation",
    "Waste": "Waste",
    "Solvents production and application": "Industry",
    "Agriculture": "AFOLU|Agriculture",
}

EARTH_RADIUS_M = 6371000


def load_gas(
    files: list[str | Path],
    gas: str,
    sector_names: dict[int, str] = SECTOR_NAMES,
    sector_mapping: dict[str, str] = SECTOR_MAPPING,
    resolution_deg: float = 0.1,
) -> xr.Dataset:
    """Load one gas: flux to mass conversion, sector mapping, annual sums.

    flux [kg/m2/s] * cell area [m2] * seconds per month / 1000 -> t/month,
    then summed to t/yr per mapped sector. Variables are named
    ``Emissions|<gas>|<sector>``; dims remain (lat, lon, year).
    """
    ds = xr.open_mfdataset(files, combine="by_coords").chunk({"lat": -1, "lon": -1})

    lat_res = np.deg2rad(resolution_deg)
    lon_res = np.deg2rad(resolution_deg)
    seconds_per_month = ds.time.dt.days_in_month * 86400 / 1000
    cell_area = (EARTH_RADIUS_M**2) * lon_res * np.abs(np.cos(np.deg2rad(ds.lat))) * lat_res
    emi_gas = ds[f"{gas}_em_anthro"] * cell_area * seconds_per_month

    # integer sector index -> CEDS name -> target sector path
    emi_gas = emi_gas.assign_coords(sector=[sector_names[i] for i in emi_gas.sector.values])
    emi_gas = emi_gas.assign_coords(sector=[sector_mapping[s] for s in emi_gas.sector.values])

    # sectors to individual variables, summing duplicates
    target_sectors = set(sector_mapping.values())
    data_vars = {}
    for target in target_sectors:
        mask = emi_gas.sector == target
        data_vars[target] = emi_gas.isel(sector=mask).sum("sector")

    ds_gas = xr.Dataset(data_vars)
    ds_gas = ds_gas.groupby("time.year").sum("time")
    ds_gas = ds_gas.rename({var: f"Emissions|{gas}|{var}" for var in ds_gas.data_vars})
    return ds_gas


def format_grid(emi: xr.Dataset) -> xr.Dataset:
    """Zeros -> NaN, (lat, lon, year) -> (y desc, x, time), float32."""
    emi = emi.where(emi != 0, np.nan)
    emi = emi.rename({"lat": "y", "lon": "x", "year": "time"})
    emi = emi.sortby("y", ascending=False)
    emi = emi.transpose("y", "x", "time")
    return emi.astype("float32")
