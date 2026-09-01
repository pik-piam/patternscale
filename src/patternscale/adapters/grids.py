"""Format-level loaders and transforms for gridded input data.

All functions take explicit file paths or already-opened datasets; directory
layouts and filename conventions are project concerns and stay outside the
package. Returned objects are raw (unharmonized); harmonize with
``patternscale.prep.harmonize`` afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def load_emissions(path: str | Path, variables: list[str]) -> xr.Dataset:
    """Base-year emission grids from zarr; '__' in names is decoded to '|'."""
    if not Path(path).exists():
        # zarr itself reports a missing store only as KeyError: '.zmetadata'
        raise FileNotFoundError(f"zarr store not found: {path}")
    emi = xr.open_zarr(str(path), consolidated=True)
    emi = emi.rename({v: v.replace("__", "|") for v in emi.data_vars})
    return emi[variables]


def classify_urbanization(urb: xr.Dataset, kind: str, threshold: float) -> xr.Dataset:
    """Classify an ``urbanization`` field into categories 10/20/30.

    Categories: 10 rural, 20 peri-urban, 30 urban.

    kind="categorical": input holds category means after coarsening;
    ``threshold`` is the fraction of fine cells with the higher category
    needed for the coarse cell to count as the higher category.

    kind="share": input holds an urban share in [0, 1]; ``threshold`` is the
    share above which a cell is classified as urban (above half of it:
    peri-urban).
    """
    if kind == "categorical":
        urb["urbanization"] = xr.where(
            (urb["urbanization"] >= 10) & (urb["urbanization"] < 10 + threshold * 10), 10,
            xr.where(
                (urb["urbanization"] >= 10 + threshold * 10) & (urb["urbanization"] < 20 + threshold * 10), 20,
                xr.where(
                    (urb["urbanization"] >= 20 + threshold * 10) & (urb["urbanization"] <= 30), 30,
                    urb["urbanization"],
                ),
            ),
        )
    elif kind == "share":
        urb["urbanization"] = xr.where(
            (urb["urbanization"] >= 0) & (urb["urbanization"] < threshold / 2), 10,
            xr.where(
                (urb["urbanization"] >= threshold / 2) & (urb["urbanization"] < threshold), 20,
                xr.where(
                    (urb["urbanization"] >= threshold) & (urb["urbanization"] <= 1), 30,
                    urb["urbanization"],
                ),
            ),
        )
    else:
        raise ValueError(f"urbanization kind '{kind}' not implemented (use 'categorical' or 'share')")

    return urb


def derive_urban_masks(urb: xr.Dataset) -> xr.Dataset:
    """Boolean urban/peri-urban masks from a classified ``urbanization`` field."""
    urb["mask_urb"] = urb["urbanization"].where(urb["urbanization"] == 30, False).astype(bool)
    urb["mask_peri"] = urb["urbanization"].where(urb["urbanization"] == 20, False).astype(bool)
    return urb


def load_region_raster(
    raster_path: str | Path,
    mapping_path: str | Path,
    drop_vars: list[str] | None = None,
    engine: str = "netcdf4",
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Region-number raster (netCDF) and region number mapping table (CSV).

    Returns (raster dataset, region map with columns Region, Region_number).
    """
    gtr = xr.open_dataset(str(raster_path), engine=engine)
    if drop_vars:
        gtr = gtr.drop_vars(drop_vars)

    region_map = pd.read_csv(mapping_path)
    region_map = region_map.rename(columns={"RegionCode": "Region", "RegionNumber": "Region_number"})
    region_map = region_map[region_map["Region"].isna() == False]  # noqa: E712
    return gtr, region_map
