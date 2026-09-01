"""Preprocessing "grid to country": country raster and region-number grids.

Input: a fine (e.g. 1-km) country raster with ISO 3166-1 numeric codes per
cell, and per region template a number mapping CSV
(``iso_lookup_numeric`` -> ``RegionNumber``).
Output:
1. a coarsened country dataset (``iso_numeric`` per cell, mode-aggregated,
   plus a numeric<->alpha-3 lookup table), and
2. one region-number raster per region template (``region_number`` per
   cell), which is the raster the engine's data contract requires.

Faithful port of ``grid-to-country.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .raster import coarsen_mode

logger = logging.getLogger(__name__)

NODATA_FALLBACK = -9999


def country_raster(tif_path: str | Path, coarse_factor: int = 12) -> xr.Dataset:
    """ISO-numeric country raster with alpha-3 lookup, mode-coarsened.

    Reads the fine country GeoTIFF, builds the numeric->alpha-3 lookup via
    pycountry, coarsens by the most frequent code per block (unless
    ``coarse_factor == 1``) and masks the 65535 nodata code.
    """
    import pycountry
    import rioxarray as rxr

    rx = rxr.open_rasterio(str(tif_path), chunks={"x": 2000, "y": 2000})
    da_iso = rx.squeeze("band", drop=True) if "band" in rx.dims else rx

    nodata = da_iso.rio.nodata
    da_iso = da_iso.astype("int32")

    # numeric -> alpha-3 lookup (small: ~200 countries)
    unique_codes = (
        np.unique(da_iso.data) if isinstance(da_iso.data, np.ndarray)
        else np.unique(da_iso.compute().data)
    )
    unique_codes = unique_codes[unique_codes != nodata] if nodata is not None else unique_codes
    unique_codes = np.asarray(sorted(int(c) for c in unique_codes if c != 0 and not np.isnan(c)))

    def numeric_to_alpha3(n):
        try:
            c = pycountry.countries.get(numeric=str(int(n)).zfill(3))
            return c.alpha_3 if c else ""
        except Exception:
            return ""

    lookup_df = pd.DataFrame({
        "iso_numeric": unique_codes.astype(np.int32),
        "iso_a3": pd.Series([numeric_to_alpha3(n) for n in unique_codes], dtype="object"),
    })
    lookup_a3_bytes = lookup_df["iso_a3"].astype(str).str.encode("ascii")

    ds = xr.Dataset(
        {"iso_numeric": da_iso.rename("iso_numeric")},
        coords={
            "x": da_iso["x"],
            "y": da_iso["y"],
            "iso_index": np.arange(len(lookup_df)),
        },
    )
    ds["iso_lookup_numeric"] = xr.DataArray(
        lookup_df["iso_numeric"].values, dims=("iso_index",),
        attrs={"description": "ISO-3166-1 numeric codes (3-digit)"},
    )
    ds["iso_lookup_a3"] = xr.DataArray(
        lookup_a3_bytes.values, dims=("iso_index",),
        attrs={"description": "ISO-3166-1 alpha-3 codes (ASCII bytes)"},
    )
    ds["iso_numeric"].attrs.update({
        "long_name": "ISO 3166-1 numeric (3-digit) country code per pixel",
        "units": "1",
        "_FillValue": int(nodata) if nodata is not None else NODATA_FALLBACK,
    })

    if coarse_factor != 1:
        ds = coarsen_mode(ds, factor=coarse_factor, var_name="iso_numeric")

    ds["iso_numeric"] = ds["iso_numeric"].astype("float32")
    ds["iso_numeric"] = ds["iso_numeric"].where(ds["iso_numeric"] != 65535)

    ds.rio.write_crs(da_iso.rio.crs, inplace=True)
    for var in ds.data_vars:
        ds[var].attrs.pop("_FillValue", None)

    return ds


def add_region_numbers(
    country_ds: xr.Dataset,
    mapping: pd.DataFrame,
    code_col: str = "iso_lookup_numeric",
    number_col: str = "RegionNumber",
) -> xr.Dataset:
    """Region-number raster from the country raster and a number mapping.

    ``mapping`` links ISO numeric codes to region numbers; unmapped cells
    become NaN. Adds ``region_number`` (float32) to a copy of the dataset.
    """
    # float values so np.vectorize infers a float output dtype regardless of
    # whether the first grid cell is NaN (the legacy code relied on that)
    number_map_dict = dict(zip(mapping[code_col], mapping[number_col].astype(float)))

    ds = country_ds.copy()
    ds["region_number"] = ds["iso_numeric"].copy()
    ds["region_number"].data = np.vectorize(lambda x: number_map_dict.get(x, np.nan))(
        ds["iso_numeric"].values
    )
    ds["region_number"] = ds["region_number"].astype("float32")
    ds["region_number"] = ds["region_number"].chunk({"y": -1, "x": -1})
    return ds


def netcdf_encoding(nodata: int | None = None) -> dict:
    """Encoding used when writing country/region rasters to netCDF."""
    return {
        "iso_numeric": {
            "dtype": "float32", "zlib": True, "complevel": 4,
            "_FillValue": int(nodata) if nodata is not None else NODATA_FALLBACK,
        },
        "iso_lookup_numeric": {"dtype": "int32"},
        "iso_lookup_a3": {"dtype": "S3"},
    }
