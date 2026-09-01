"""Result container and output writers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import xarray as xr


def collect_data(data_list: list[xr.DataArray]) -> pd.DataFrame:
    """Concatenate named per-region time series into a long DataFrame.

    Each DataArray carries the region label as its ``name``; the result has
    columns (year, value, region).
    """
    dfs = []
    for da in data_list:
        df = da.to_dataframe(name="value").reset_index()
        df["region"] = da.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.rename(columns={"time": "year"})
    return out


@dataclass
class DownscalingResults:
    """Output of one downscaling run.

    - ``table``: wide DataFrame with index columns
      (model, scenario, variable, region) and one column per year. Contains
      regional sums of the downscaled variables, mask-restricted sums and the
      normalization factors (``<var>|NormFactor``, ``<var>|NormFactor_EmiCap``).
    - ``grid``: Dataset with one data variable per target variable
      (``VariableSpec.short``), dims (time, y, x), values in grid units.
    """

    table: pd.DataFrame
    grid: xr.Dataset
    meta: dict = field(default_factory=dict)

    def table_long(self) -> pd.DataFrame:
        """Long-format table: (model, scenario, variable, region, year, value).

        Canonical format for programmatic use — unlike the wide CSV, it
        round-trips through CSV without the year columns changing type.
        """
        return self.table.melt(
            id_vars=["model", "scenario", "variable", "region"],
            var_name="year",
            value_name="value",
        )

    def save_table(self, path: str | Path, format: str = "wide") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "wide":
            self.table.to_csv(path, index=False)
        elif format == "long":
            self.table_long().to_csv(path, index=False)
        else:
            raise ValueError(f"unknown table format '{format}' (use 'wide' or 'long')")

    def save_grid(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.to_zarr(path, mode="w", consolidated=True)
