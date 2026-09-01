"""Shared low-level readers, transforms and writers for preprocessing.

Ports of the legacy ``import_dir``, ``load_netcdf_by_gas`` and the
mode-coarsening path of ``coarsen_save_rio_xarray``, plus the recurring
"mask zeros, chunk, write zarr" output block of the process notebooks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def import_dir(
    data_dir: str | Path, varname: str, glob_pattern: str, search_pattern: str
) -> xr.Dataset:
    """Stack a directory of single-year GeoTIFFs into one (time, y, x) dataset.

    ``search_pattern`` must contain one capture group for the year. Zeros are
    masked to NaN.
    """
    import rioxarray as rxr

    files = sorted(Path(data_dir).glob(glob_pattern))
    data_list = []
    years = []
    for f in files:
        year = int(re.search(search_pattern, f.name).group(1))
        years.append(year)
        da = rxr.open_rasterio(f, chunks={"x": -1, "y": -1}, masked=True)
        data_list.append(da)
    ds_rxr = xr.concat(data_list, dim="time")
    ds_rxr = ds_rxr.assign_coords(time=years)
    ds_rxr = ds_rxr.to_dataset(name=varname)
    ds = ds_rxr.where(ds_rxr != 0, np.nan).squeeze(drop=True)
    return ds


def load_netcdf_by_gas(
    file_paths: list[str | Path],
    years: list[int] | None = None,
) -> dict[str, xr.Dataset]:
    """Group per-sector netCDF files by gas into (year, y, x) datasets.

    Filenames must match ``*_<gas>_<year>_<sector>_*.nc``. The latitude axis
    is renamed to ``y``/``x`` and flipped to descending ``y``.
    """
    pattern = re.compile(r".+_(?P<gas>.+)_(?P<year>\d{4})_(?P<sector>.+)_[^_]+\.nc$")

    grouped: dict[str, dict[int, dict[str, Path]]] = {}

    for fp in file_paths:
        m = pattern.match(str(Path(fp).name))
        if not m:
            raise ValueError(f"Filename does not match expected pattern: {fp}")
        gas = m.group("gas")
        year = int(m.group("year"))
        sector = m.group("sector")

        if years is not None and year not in years:
            continue

        if gas not in grouped:
            grouped[gas] = {}
        if year not in grouped[gas]:
            grouped[gas][year] = {}
        grouped[gas][year][sector] = Path(fp)

    datasets: dict[str, xr.Dataset] = {}

    for gas, years_dict in grouped.items():
        yearly_datasets = []

        for year in sorted(years_dict):
            sector_arrays = {}

            for sector, path in years_dict[year].items():
                da = xr.open_dataset(path)
                var_name = list(da.data_vars)[0]
                sector_arrays[sector] = da[var_name]

            year_ds = xr.Dataset(sector_arrays)
            year_ds = year_ds.expand_dims({"year": [year]})
            yearly_datasets.append(year_ds)

        # rename and mirror on equator
        ds_gas = xr.concat(yearly_datasets, dim="year")
        ds_gas = ds_gas.rename({"lat": "y", "lon": "x"})
        ds_gas = ds_gas.isel(y=slice(None, None, -1))

        datasets[gas] = ds_gas

    return datasets


def average_base_years(
    ds: xr.Dataset | xr.DataArray,
    base_years: list[int],
    window: int = 2,
    dim: str = "year",
    out_dim: str | None = None,
    clip_to_available: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Centered multi-year means around each base year.

    ``window=2`` gives 5-year means. With ``clip_to_available`` the window is
    reduced to the years present in the data (e.g. 2018-2022 for base year
    2020 when the data ends in 2022); otherwise missing years raise.
    """
    out_dim = out_dim or dim
    if clip_to_available:
        avail = set(np.asarray(ds[dim].values).tolist())
        slices = [
            ds.sel({dim: [y + d for d in range(-window, window + 1) if (y + d) in avail]}).mean(dim)
            for y in base_years
        ]
    else:
        slices = [ds.sel({dim: range(y - window, y + window + 1)}).mean(dim) for y in base_years]

    return xr.concat(slices, dim=pd.Index(base_years, name=out_dim))


def coarsen_mode(ds: xr.Dataset, factor: int, var_name: str) -> xr.Dataset:
    """Coarsen a categorical raster by the most frequent value per block.

    Port of the "mode" path of the legacy ``coarsen_save_rio_xarray``:
    coarsen with boundary="trim", auto-rechunk, and update the affine
    transform/CRS so the result stays georeferenced.
    """
    from affine import Affine
    from scipy.stats import mode

    da = ds[var_name]

    def find_mode(arr, axis):
        m = mode(arr, axis=axis)
        return np.atleast_2d(m[0])

    ds_coarse = ds.coarsen(y=factor, x=factor, boundary="trim").reduce(find_mode)
    ds_coarse = ds_coarse.chunk({"y": "auto", "x": "auto"})

    old_transform = ds.rio.transform()
    new_transform = Affine(
        old_transform.a * factor, old_transform.b, old_transform.c,
        old_transform.d, old_transform.e * factor, old_transform.f,
    )
    ds_coarse = ds_coarse.rio.write_transform(new_transform)

    if da.rio.crs is not None:
        ds_coarse[var_name] = (
            ds_coarse[var_name].rio.write_crs(da.rio.crs).rio.write_nodata(da.rio.nodata)
        )

    logger.info(f"coarsen_mode: {ds[var_name].shape} -> {ds_coarse[var_name].shape} (factor {factor})")
    return ds_coarse


def mask_zeros(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Replace zeros with NaN (lets zarr skip empty chunks)."""
    return ds.where(ds != 0, np.nan)


def encode_pipe_names(ds: xr.Dataset) -> xr.Dataset:
    """Encode '|' in variable names as '__' for storage (decoded on load)."""
    return ds.rename({v: v.replace("|", "__") for v in ds.data_vars})


def save_zarr(ds: xr.Dataset, path: str | Path, chunks: dict) -> None:
    """Write a dataset to a (consolidated) zarr store, overwriting."""
    ds = ds.chunk(chunks)
    ds.to_zarr(
        str(path),
        mode="w",  # deletes/overwrites older results
        consolidated=True,
        write_empty_chunks=False,
    )
    logger.info(f"saved: {path}")
