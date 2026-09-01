"""Preprocessing of EDGAR annual sectoral emission grids.

Input: per-sector netCDF files (``EDGAR_2025_GHG_<gas>_<year>_<sector>_*.nc``)
in t/yr on a 0.1-degree grid, plus a variable mapping table
(EDGAR sector -> IAMC-style sector path).
Output: emission grids with variable names ``Emissions|<gas>|<sector>``.

Faithful port of ``process_emissions_allOfEDGAR.ipynb``, including the
NaN-safe sector accumulation: plain ``+`` would propagate NaN and silently
drop all emissions in cells where the summed sector source patterns do not
overlap (e.g. power plants vs oil/gas fields cost Energy|Supply ~80% of CH4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def read_sector_mapping(path: str | Path) -> dict[str, str]:
    """EDGAR sector -> IAMC-style path from a variablemapping CSV.

    The CSV needs columns ``EDGAR`` and ``Variable``; rows without a
    ``Variable`` entry are dropped (those sectors are ignored).
    """
    varmap = pd.read_csv(path)
    varmap = varmap[~varmap["Variable"].isna()]
    return varmap.set_index("EDGAR")["Variable"].to_dict()


def aggregate_sectors(
    emi_by_gas: dict[str, xr.Dataset],
    sector_mapping: dict[str, str],
    target_paths: list[str] | None = None,
) -> xr.Dataset:
    """Map EDGAR sectors to target variables, NaN-safe summing collisions.

    ``emi_by_gas``: per-gas datasets with one variable per EDGAR sector
    (as returned by ``raster.load_netcdf_by_gas``). ``target_paths``
    optionally restricts the output to a subset of sector paths.
    Zeros are re-masked to NaN after summing.
    """
    all_vars: dict[str, xr.DataArray] = {}
    for gas, ds in emi_by_gas.items():
        aggregated: dict[str, xr.DataArray] = {}

        for sector in ds.data_vars:
            path = sector_mapping.get(sector)
            if path is None:
                continue
            if target_paths is not None and path not in target_paths:
                continue
            out_name = f"Emissions|{gas}|{path}"
            # NaN-safe accumulation (see module docstring)
            if out_name in aggregated:
                aggregated[out_name] = aggregated[out_name].fillna(0) + ds[sector].fillna(0)
            else:
                aggregated[out_name] = ds[sector]

        # re-mask zeros as NaN after summing
        for k, v in aggregated.items():
            aggregated[k] = v.where(v != 0)
        all_vars.update(aggregated)

    return xr.Dataset(all_vars)


def add_derived_variables(
    emi: xr.Dataset, gases: list[str] = ("CO2", "CH4", "N2O", "GHG", "CO2bio")
) -> xr.Dataset:
    """Industry total and total excluding international bunkers, per gas.

    Industry = combustion for manufacturing (Energy|Demand|Industry)
             + Industrial Processes.
    """
    for gas in gases:
        emi[f"Emissions|{gas}|Industry"] = (
            emi[f"Emissions|{gas}|Energy|Demand|Industry"].fillna(0)
            + emi[f"Emissions|{gas}|Industrial Processes"].fillna(0)
        )
        emi[f"Emissions|{gas}|Industry"] = emi[f"Emissions|{gas}|Industry"].where(
            emi[f"Emissions|{gas}|Industry"] != 0
        )

        emi[f"Emissions|{gas}|Total_excl_bunkers"] = (
            emi[f"Emissions|{gas}|Total"].fillna(0)
            - emi[f"Emissions|{gas}|Transportation|Pass|Aviation|International|Demand"].fillna(0)
            - emi[f"Emissions|{gas}|Transportation|Freight|International Shipping|Demand"].fillna(0)
        )
        emi[f"Emissions|{gas}|Total_excl_bunkers"] = emi[
            f"Emissions|{gas}|Total_excl_bunkers"
        ].where(emi[f"Emissions|{gas}|Total_excl_bunkers"] != 0)

    return emi


def combine_industry(aggregated: xr.Dataset, gases: list[str]) -> xr.Dataset:
    """Replace Energy|Demand|Industry + Industrial Processes by Industry.

    Used for the annual grids where only the downscaling sectors are kept.
    """
    for gas in gases:
        aggregated[f"Emissions|{gas}|Industry"] = (
            aggregated[f"Emissions|{gas}|Energy|Demand|Industry"].fillna(0)
            + aggregated[f"Emissions|{gas}|Industrial Processes"].fillna(0)
        )
        aggregated[f"Emissions|{gas}|Industry"] = aggregated[f"Emissions|{gas}|Industry"].where(
            aggregated[f"Emissions|{gas}|Industry"] != 0
        )
        aggregated = aggregated.drop_vars(
            [f"Emissions|{gas}|Energy|Demand|Industry", f"Emissions|{gas}|Industrial Processes"]
        )
    return aggregated
