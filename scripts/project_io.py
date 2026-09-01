"""Data loading for the city-downscaling runs: paths and filename conventions.

All loaders take the data directories explicitly:
- ``processed``: preprocessed input stores (zarr) and scenario CSVs
- ``mappings`` : region rasters and mapping tables

wrapping the format-level loaders of ``patternscale.adapters``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

import patternscale as ps


def _open_zarr(path: str) -> xr.Dataset:
    """Open a zarr store, failing clearly if the path does not exist.

    zarr itself reports a missing store only as ``KeyError: '.zmetadata'``,
    which is hard to diagnose (case-sensitive filesystems!).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"zarr store not found: {path}")
    return xr.open_zarr(path, consolidated=True)


def load_population(processed: Path, source: str, ssp: str, resolution_km: int) -> xr.Dataset:
    return _open_zarr(f"{processed}/Population_{source}_{ssp}_cf_{resolution_km}.zarr")


def load_gdp(processed: Path, source: str, ssp: str, resolution_km: int) -> xr.Dataset:
    return _open_zarr(f"{processed}/GDP_PPP_{source}_{ssp}_cf_{resolution_km}.zarr")


def load_typology(processed: Path, source: str, resolution_km: int) -> xr.Dataset:
    return _open_zarr(f"{processed}/typology_{source}_cf_{resolution_km}.zarr")


def load_emissions(processed: Path, source: str, resolution_km: int, variables: list[str]) -> xr.Dataset:
    return ps.grids.load_emissions(
        f"{processed}/Emissions_all_{source}_cf_{resolution_km}.zarr", variables
    )


def load_urbanization(
    processed: Path, source: str, ssp: str, years: list[int], threshold: float | None = None
) -> xr.Dataset:
    """Urbanization masks / shares by source.

    'Diego' and 'Compass_bua' are classified into categories and boolean
    masks; 'Compass_pop' already provides urban/peri-urban population shares
    (mask_urb, mask_peri).
    """
    if source == "Diego":
        # only works for 12 km resolution (cf 10)
        urb = _open_zarr(f"{processed}/urbanization_Diego_SSP2_cf_10.zarr")
        urb = ps.grids.classify_urbanization(urb, kind="categorical", threshold=threshold)
        urb = urb.expand_dims(time=years)
        urb = ps.grids.derive_urban_masks(urb)
    elif source == "Compass_bua":
        # only works for 12 km resolution, SSP2
        urb = _open_zarr(f"{processed}/urbanization_bua_Compass_SSP2_cf_12.zarr")
        urb = ps.grids.classify_urbanization(urb, kind="share", threshold=threshold)
        urb = ps.grids.derive_urban_masks(urb)
    elif source == "Compass_pop":
        urb = _open_zarr(f"{processed}/urbanization_pop_Compass_{ssp}_cf_12.zarr")
    else:
        raise ValueError(f"urbanization source '{source}' not implemented")

    return urb


def load_region_raster(
    mappings: Path, template_regions: str, resolution_km: int
) -> tuple[xr.Dataset, pd.DataFrame]:
    # SMIP country templates of any version (SMIP_countries[_v1-1-x]) and REMIND
    # share the same file naming; matched case-insensitively because the
    # on-disk casing has varied between versions
    key = template_regions.lower()
    if key.startswith("smip_countries") or key == "remind":
        return ps.grids.load_region_raster(
            f"{mappings}/{template_regions}_regions_{resolution_km}km.nc",
            f"{mappings}/numbermapping_{template_regions}.csv",
        )
    if template_regions == "IMAGE":
        return ps.grids.load_region_raster(
            f"{mappings}/IMAGE_GADM_regions_raster.nc",
            f"{mappings}/numbermapping_{template_regions}.csv",
            drop_vars=["band", "country_id_GADM"],
        )
    raise ValueError(f"template_regions '{template_regions}' not implemented")


def load_smip_scenario(
    processed: Path, scenario_name: str, version: str, target_variables: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    return ps.smip.load_scenario(
        f"{processed}/IAM_smip_country_{scenario_name}_{version}.csv", target_variables
    )
