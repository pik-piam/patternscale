"""Preprocessing of the city-typology raster.

Input: one GeoTIFF with integer typology classes (1-4) per cell.
Output: dataset with one variable per class ("Type 1" ... "Type 4"),
True where the cell belongs to the class and NaN elsewhere.

Faithful port of ``process_typology.ipynb``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def load_typology(path: str | Path, n_types: int = 4, varname: str = "Typology") -> xr.Dataset:
    """Typology class raster -> one True/NaN mask variable per class."""
    import rioxarray as rxr

    typ = rxr.open_rasterio(str(path), chunks={"x": -1, "y": -1}, masked=True)
    typ = typ.squeeze()  # remove band dimension

    ds = typ.to_dataset(name=varname)
    for i in range(1, n_types + 1):
        ds[f"Type {i}"] = xr.where(ds[varname] == i, True, np.nan)

    ds = ds.drop_vars(varname)
    ds = ds.reset_coords(["band", "spatial_ref"], drop=True)
    return ds
