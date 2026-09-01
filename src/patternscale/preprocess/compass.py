"""Preprocessing of COMPASS 6-arcmin GeoTIFF projections.

Input: per-year GeoTIFF stacks (historical, 2025 not-harmonized, SSP
projection not-harmonized) for GDP, population counts, degree of
urbanization (built-up-area share) and city/town population counts.
Output: (time, y, x) datasets; urban/peri-urban population shares are
derived from the city/town population counts.

Faithful port of ``process_compass.ipynb``. The 6-arcmin source resolution
equals the 0.1-degree target grid, so no coarsening is involved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from .raster import import_dir

logger = logging.getLogger(__name__)


def load_projection_stack(
    data_dir: str | Path, varname: str, stem: str, ssp: str
) -> xr.Dataset:
    """Concatenate the three COMPASS file generations along time.

    - ``<stem>_<year>_6min.tif``                : harmonized historical years
    - ``<stem>_<year>_6min_not_harm.tif``       : 2025 (not harmonized)
    - ``<stem>_<year>_6min_<SSP>_not_harm.tif`` : SSP projection years
    """
    hist = import_dir(
        data_dir, varname,
        f"{stem}_????_6min.tif",
        f"{stem}_(\\d{{4}})_6min.tif",
    )
    y2025 = import_dir(
        data_dir, varname,
        f"{stem}_????_6min_not_harm.tif",
        f"{stem}_(\\d{{4}})_6min_not_harm.tif",
    )
    y2025 = y2025.assign_coords(time=2025)
    proj = import_dir(
        data_dir, varname,
        f"{stem}_????_6min_{ssp}_not_harm.tif",
        f"{stem}_(\\d{{4}})_6min_{ssp}_not_harm.tif",
    )
    return xr.concat([hist, y2025, proj], dim="time")


def pop_share_masks(
    urb_city: xr.Dataset, urb_town: xr.Dataset, pop: xr.Dataset,
    pop_var: str = "Population",
) -> xr.Dataset:
    """Urban/peri-urban population shares from city/town population counts.

    ``urb_city``/``urb_town`` hold the counts as ``mask_urb``/``mask_peri``
    (cities = urban, towns = peri-urban); dividing by total population turns
    them into shares.
    """
    urb_city = urb_city.copy()
    urb_town = urb_town.copy()
    urb_city["mask_urb"] = urb_city["mask_urb"] / pop[pop_var]
    urb_town["mask_peri"] = urb_town["mask_peri"] / pop[pop_var]
    return xr.merge([urb_city, urb_town]).reset_coords("spatial_ref", drop=True)
