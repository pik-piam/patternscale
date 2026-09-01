"""Auxiliary regional aggregation helpers.

Post-processing (shares, intensities, GHG conversion, region re-aggregation)
is outside the package scope. This module only provides the generic building
block "regional (optionally mask-restricted) sums of a gridded field", which
project code can use e.g. to aggregate proxy grids (urban population, ...)
consistently with the pipeline's own sums.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

from .results import collect_data


def regional_mask_sums(
    da: xr.DataArray,
    region_number: xr.DataArray,
    region_map: pd.DataFrame,
    mask: xr.DataArray | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Sum a gridded field per region, optionally restricted to a mask.

    Returns a long DataFrame with columns (year, value, region); values are
    divided by ``scale``.
    """
    field = da if mask is None else (da * mask)
    try:
        field = field.persist()
    except AttributeError:
        pass

    data = []
    for R in region_map["Region_number"].unique():
        reg = region_map.loc[region_map["Region_number"] == R, "Region"].unique()[0]
        region_sum = (field * (region_number == R)).sum(dim=["x", "y"], skipna=True) / scale
        region_sum.name = reg
        data.append(region_sum)

    return collect_data(data)
