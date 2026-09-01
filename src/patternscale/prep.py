"""Preparation of input grids: harmonization and pre-downscaling corrections.

Harmonization generalizes the legacy ``harmonize_datasets``: coordinates are
replaced by the clean coordinate arrays of the target ``Grid`` via
nearest-neighbour reindexing (tolerance = half a cell), removing
floating-point misalignments between sources. Datasets may be dask-backed;
all operations stay lazy except the optional checksum.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from .grid import Grid

logger = logging.getLogger(__name__)


def harmonize_dataset(
    ds: xr.Dataset,
    grid: Grid,
    years: list[int] | None = None,
    check_sum: bool = True,
) -> xr.Dataset:
    """Reindex a dataset onto the clean target grid.

    If ``years`` is given, the time axis is renamed/reindexed to exactly these
    steps; otherwise the dataset is treated as time-independent.
    ``check_sum`` verifies that no data was lost by the reindexing (exact
    comparison of sums, first year only for time-dependent data).
    """
    x_clean = grid.x_coords()
    y_clean = grid.y_coords()

    # some input data objects might still use "year"
    if "year" in ds.coords:
        ds = ds.rename({"year": "time"})

    if "time" in ds.dims:
        ds = ds.transpose("y", "x", "time")

    if "spatial_ref" in ds.coords:
        ds = ds.reset_coords("spatial_ref", drop=True)
    if "spatial_ref" in ds.data_vars:
        ds = ds.drop_vars("spatial_ref")

    if years is not None:
        ds_new = ds.reindex(
            y=y_clean,
            x=x_clean,
            time=years,
            fill_value=False,
            method="nearest",
            tolerance=grid.reindex_tolerance,
        )
    else:
        ds_new = ds.reindex(
            y=y_clean,
            x=x_clean,
            fill_value=False,
            method="nearest",
            tolerance=grid.reindex_tolerance,
        )

    if check_sum:
        for var in ds.data_vars:
            if years is not None:
                diff = (
                    ds[var].sel(time=years[0]).sum()
                    - ds_new[var].sel(time=years[0]).sum()
                ).compute()
            else:
                diff = (ds[var].sum() - ds_new[var].sum()).compute()
            if diff != 0:
                raise ValueError(
                    f"Data lost on reindexing for variable {var}. Difference: {diff}"
                )

    return ds_new


def apply_min_proxy(
    da: xr.DataArray,
    min_value: float,
    grid: Grid,
    log_diagnostics: bool = False,
) -> xr.DataArray:
    """Floor non-empty proxy cells at ``min_value`` per 100 km^2.

    Prevents near-empty cells from dominating relative-growth dynamics. Empty
    (NaN) cells stay empty; the floor is scaled by latitude-dependent cell
    area. The array is promoted to float64 (legacy behaviour).
    """
    da = da.astype("float64")
    mask = da.where(pd.notnull(da), np.nan)

    cell_area = grid.cell_area_per_100km2(da["y"])
    da_min = xr.where(
        mask, xr.where(da < min_value * cell_area, min_value * cell_area, da), np.nan
    )

    if log_diagnostics:
        created = (da_min.sum() - da.sum()).data
        try:
            created = created.compute()
        except AttributeError:
            pass
        logger.info(f"{da.name} created over all years by min-proxy floor: {created}")

    return da_min
