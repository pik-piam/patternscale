r"""Core computation steps of the pattern-scaling method.

Contents (in pipeline order):

- region utilities: broadcasting regional values onto the grid
- the scaling kernel:

  .. math::

      E_{t,g} = E_{t_0,g} \cdot \frac{E_{t,R}}{E_{t_0,R}}
                \cdot \frac{P_{t,g}/P_{t_0,g}}{P_{t,R}/P_{t_0,R}}

- normalization of regional grid sums to scenario totals (two points per
  region and variable: directly after the kernel, and after intensity-based
  corrections; with ``skip_first_norm=True`` the first factor is 1 and the
  renormalization targets the scenario total directly — legacy default)
- optional corrections; currently ``new_dev`` ("newly developed cells"):
  cells whose relative proxy growth exceeds the regional relative proxy
  growth by more than a threshold are set to the regional mean emission
  intensity (scenario E/P), followed by a renormalization. Further
  corrections (e.g. hard intensity capping) follow the same pattern: a
  mask/preparation step at variable level and an application step at region
  level.

All functions are pure and operate on already-aligned xarray/pandas objects
(data contract, see ``patternscale.contract``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# region utilities
# ---------------------------------------------------------------------------


def map_to_grid(da: xr.DataArray, region_number: xr.DataArray) -> xr.DataArray:
    """Broadcast a (time, Region_number) array onto the (time, y, x) grid.

    Every grid cell receives the value of its region; cells whose region
    number does not appear in ``da`` become NaN. Equivalent to (but much
    faster than) ``da.where(da.Region_number == region_number).sum(...)``.
    """
    region_map_np = np.nan_to_num(region_number.values, nan=0).astype(int)
    max_r = int(region_map_np.max())
    lookup = np.full(max_r + 1, -1, dtype=int)
    for i, rn in enumerate(da.Region_number.values):
        if 0 <= int(rn) <= max_r:
            lookup[int(rn)] = i
    grid_idx = lookup[region_map_np]
    valid = grid_idx >= 0
    data = da.values[:, np.where(valid, grid_idx, 0)]
    data[:, ~valid] = np.nan
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": da.time, "y": region_number.y, "x": region_number.x},
    )


# ---------------------------------------------------------------------------
# scaling kernel
# ---------------------------------------------------------------------------


@dataclass
class ScalingResult:
    """Output of the kernel plus intermediates needed by corrections."""

    E_t: xr.DataArray          # unnormalized gridded trajectory (time, y, x)
    P_g_rel: xr.DataArray      # gridded proxy relative to base year
    P_R_rel_regi: xr.DataArray # regional relative proxy, broadcast to grid


def relative_to_base(df: pd.DataFrame, base_year: int) -> xr.DataArray:
    """Regional values relative to the base year as (time, Region_number).

    ``df`` holds one scenario variable with columns Year, Value,
    Region_number.
    """
    rel = df[["Year", "Value", "Region_number"]].copy()
    base = rel[rel["Year"] == base_year].set_index("Region_number")["Value"]
    rel["Value_rel"] = rel["Value"] / rel["Region_number"].map(base)
    da = rel.set_index(["Year", "Region_number"])["Value_rel"].to_xarray()
    return da.rename({"Year": "time"})


def scale_variable(
    E_t0_g: xr.DataArray,
    P_t_g: xr.DataArray,
    E_t_R: pd.DataFrame,
    P_t_R: pd.DataFrame,
    region_number: xr.DataArray,
    base_year: int,
) -> ScalingResult:
    """Apply the pattern-scaling kernel for one variable.

    Parameters
    ----------
    E_t0_g : base-year grid pattern of the target variable (y, x)
    P_t_g : gridded proxy for all years (time, y, x)
    E_t_R : regional scenario data of the target variable (long df)
    P_t_R : regional scenario data of the proxy (long df)
    region_number : region raster (y, x)
    base_year : reference year t0

    Returns a ``ScalingResult`` with the unnormalized gridded trajectory
    (time, y, x) and the proxy intermediates.
    """
    P_t0_g = P_t_g.sel(time=base_year)
    P_g_rel = P_t_g / P_t0_g

    E_R_rel = relative_to_base(E_t_R, base_year)
    P_R_rel = relative_to_base(P_t_R, base_year)

    E_R_rel_regi = map_to_grid(E_R_rel, region_number)
    P_R_rel_regi = map_to_grid(P_R_rel, region_number)

    E_t = E_t0_g * E_R_rel_regi * P_g_rel / P_R_rel_regi.where(P_R_rel_regi != 0)
    return ScalingResult(E_t=E_t, P_g_rel=P_g_rel, P_R_rel_regi=P_R_rel_regi)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def first_norm_factor(
    IAM_R: xr.DataArray, sum_grid_R: xr.DataArray, skip_first_norm: bool
) -> xr.DataArray:
    """Factor normalizing regional grid sums to the scenario total.

    ``IAM_R``: regional scenario totals over time (scenario units).
    ``sum_grid_R``: regional grid sums over time (scenario units, zeros
    replaced by NaN).
    """
    if skip_first_norm:
        return xr.ones_like(IAM_R)
    return IAM_R / sum_grid_R


def renorm_factor(
    IAM_R: xr.DataArray,
    emi_before,
    emi_after,
    skip_first_norm: bool,
    grid_to_scenario_factor: float,
) -> xr.DataArray:
    """Renormalization factor after an intensity-based correction.

    ``emi_before``/``emi_after`` are regional grid sums (grid units) before
    and after the correction.
    """
    if skip_first_norm:
        return IAM_R * grid_to_scenario_factor / emi_after
    return emi_before / emi_after


# ---------------------------------------------------------------------------
# corrections
# ---------------------------------------------------------------------------

INTENSITY_MERGE_KEYS = ["Model", "Scenario", "Region", "Year", "Region_number"]


def regional_intensity(
    P_t_R: pd.DataFrame, E_t_R: pd.DataFrame, intensity_divisor: float
) -> pd.DataFrame:
    """Regional scenario emission intensity E/P per region and year.

    Returns the merged frame with the intensity in column ``Value``
    (grid units per grid-proxy unit, via ``intensity_divisor``).
    """
    keys = [k for k in INTENSITY_MERGE_KEYS if k in P_t_R.columns and k in E_t_R.columns]
    merged = pd.merge(P_t_R, E_t_R, on=keys)
    merged["Value"] = merged["Value_y"] / merged["Value_x"] / intensity_divisor
    return merged


def new_dev_mask(
    P_g_rel: xr.DataArray, P_R_rel_regi: xr.DataArray, threshold: float
) -> xr.DataArray:
    """Cells growing faster than ``threshold`` times their region's proxy."""
    return xr.where((P_g_rel / P_R_rel_regi) > threshold, True, False)


def intensity_to_da(intensity_df: pd.DataFrame, region_number: int) -> xr.DataArray:
    """Regional intensity trajectory of one region as a (time,) DataArray."""
    IAM_int = intensity_df[intensity_df["Region_number"] == region_number]
    da = IAM_int[["Value", "Year"]].set_index(IAM_int["Year"]).to_xarray()
    da = da.rename({"Year": "time"})
    return da["Value"]


def apply_new_dev_region(
    E_norm_R: xr.DataArray,
    P_t_g: xr.DataArray,
    IAM_int_da: xr.DataArray,
    nd_mask: xr.DataArray,
    mask_R: xr.DataArray,
) -> xr.DataArray:
    """Set newly developed cells of one region to the regional intensity.

    Cells with proxy <= 0 become NaN; flagged cells (including previously
    empty ones) receive ``intensity * proxy``.
    """
    prox_mask = P_t_g > 0
    E_int = xr.where(prox_mask, E_norm_R / P_t_g, np.nan)
    E_int_newDev = xr.where(nd_mask, IAM_int_da, E_int).where(mask_R)
    return xr.where(prox_mask, E_int_newDev * P_t_g, np.nan)
