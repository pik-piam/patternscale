"""Input data contract of the patternscale engine.

The engine consumes exactly three objects; everything upstream of this
contract (file formats, source-specific fixes) belongs to adapters or project
code.

1. Scenario data (``pandas.DataFrame``, long format), required columns:
   - ``Variable``      : str, must cover every ``VariableSpec.name`` and every
                         ``ProxySpec.scenario_var``
   - ``Region``        : str, region label
   - ``Region_number`` : int, links regions to the grid's region raster
   - ``Year``          : int
   - ``Value``         : float, regional total in scenario units
   Optional passthrough columns: ``Model``, ``Scenario``, ``Unit``.

2. Grid data (``xarray.Dataset``) on the target grid with dims ``y, x`` and
   for time-dependent variables ``time``:
   - one data variable per ``VariableSpec.name`` containing the base-year
     pattern (``time`` must include the base year),
   - one data variable per ``ProxySpec.grid_var`` with all downscaling years,
   - one 2-D region raster (``DownscalingConfig.region_var``) holding the
     region number of each cell (NaN outside regions).

3. Region map (``pandas.DataFrame``) with columns ``Region`` and
   ``Region_number``: the set of regions to downscale. Regions present in the
   grid raster but not in the map are ignored.

``validate_inputs`` checks the full contract and raises ``ContractError``
listing all violations at once.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from .config import DownscalingConfig

SCENARIO_REQUIRED_COLUMNS = ["Variable", "Region", "Region_number", "Year", "Value"]


class ContractError(ValueError):
    """Raised when input data violates the patternscale data contract."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        msg = "Input data violates the patternscale data contract:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        super().__init__(msg)


def scenario_variables(cfg: DownscalingConfig) -> list[str]:
    """All scenario variable names the engine needs."""
    target = [v.name for v in cfg.variables]
    proxy = [p.scenario_var for p in cfg.proxies.values()]
    return target + sorted(set(proxy) - set(target))


def validate_scenario_data(df: pd.DataFrame, cfg: DownscalingConfig) -> list[str]:
    problems: list[str] = []

    missing_cols = [c for c in SCENARIO_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"scenario data misses required columns: {missing_cols}")
        return problems  # further checks would fail

    if df["Region_number"].isna().any():
        problems.append("scenario data contains NaN Region_number entries")

    dups = df[df.duplicated(["Variable", "Region", "Year"])]
    if len(dups) > 0:
        examples = dups[["Variable", "Region", "Year"]].head(5).to_dict("records")
        problems.append(f"scenario data contains duplicates, e.g. {examples}")

    needed = scenario_variables(cfg)
    present = set(df["Variable"].unique())
    missing_vars = [v for v in needed if v not in present]
    if missing_vars:
        problems.append(f"scenario data misses variables: {missing_vars}")

    # every needed variable must exist for the base year and every target year
    years = set(cfg.years)
    for var in needed:
        if var in missing_vars:
            continue
        var_years = set(df.loc[df["Variable"] == var, "Year"].unique())
        missing_years = sorted(years - var_years)
        if missing_years:
            problems.append(f"variable '{var}' misses years {missing_years}")

    return problems


def validate_grid_data(ds: xr.Dataset, cfg: DownscalingConfig) -> list[str]:
    problems: list[str] = []

    for dim in ("y", "x"):
        if dim not in ds.dims:
            problems.append(f"grid dataset misses dimension '{dim}'")
    if problems:
        return problems

    if cfg.region_var not in ds:
        problems.append(f"grid dataset misses region raster '{cfg.region_var}'")
    else:
        region = ds[cfg.region_var]
        if set(region.dims) != {"y", "x"}:
            problems.append(
                f"region raster '{cfg.region_var}' must have dims (y, x), got {region.dims}"
            )

    def _check_var(name: str, required_years: list[int]) -> None:
        if name not in ds:
            problems.append(f"grid dataset misses variable '{name}'")
            return
        da = ds[name]
        if "time" not in da.dims:
            problems.append(f"grid variable '{name}' misses 'time' dimension")
            return
        have = set(np.asarray(da["time"].values).tolist())
        missing = [t for t in required_years if t not in have]
        if missing:
            problems.append(f"grid variable '{name}' misses time steps {missing}")

    for v in cfg.variables:
        _check_var(v.name, [cfg.base_year])
    for p in cfg.proxies.values():
        _check_var(p.grid_var, list(cfg.years))

    return problems


def validate_region_map(region_map: pd.DataFrame, ds: xr.Dataset | None, cfg: DownscalingConfig) -> list[str]:
    problems: list[str] = []
    for col in ("Region", "Region_number"):
        if col not in region_map.columns:
            problems.append(f"region map misses column '{col}'")
    if problems:
        return problems

    if region_map["Region_number"].isna().any():
        problems.append("region map contains NaN Region_number entries")
    dups = region_map[region_map.duplicated("Region_number")]
    if len(dups) > 0:
        problems.append(
            f"region map contains duplicated Region_number entries: "
            f"{sorted(dups['Region_number'].unique().tolist())}"
        )

    if ds is not None and cfg.region_var in ds:
        raster_numbers = ds[cfg.region_var].values
        raster_numbers = set(
            np.unique(raster_numbers[~np.isnan(raster_numbers)]).astype(int).tolist()
        )
        not_in_raster = sorted(
            set(region_map["Region_number"].astype(int)) - raster_numbers
        )
        if not_in_raster:
            # tolerated (legacy behaviour): such regions yield NaN sums
            logging.getLogger(__name__).warning(
                f"Region map regions without any grid cell in the region raster: {not_in_raster}"
            )

    return problems


def validate_inputs(
    scenario_df: pd.DataFrame,
    ds: xr.Dataset,
    region_map: pd.DataFrame,
    cfg: DownscalingConfig,
    strict: bool = True,
) -> list[str]:
    """Validate all engine inputs against the data contract.

    Returns the list of problems. If ``strict`` (default), raises
    ``ContractError`` when problems are found.
    """
    problems = (
        validate_scenario_data(scenario_df, cfg)
        + validate_grid_data(ds, cfg)
        + validate_region_map(region_map, ds, cfg)
    )
    if strict and problems:
        raise ContractError(problems)
    return problems


def restrict_to_complete_regions(
    scenario_df: pd.DataFrame,
    region_map: pd.DataFrame,
    cfg: DownscalingConfig,
    variables: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop regions that do not report every needed scenario variable.

    Returns the filtered scenario data, the filtered region map and the list
    of dropped region labels (matching the legacy 'regions_with_all_vars'
    filtering). ``variables`` overrides the variable set used for the
    completeness check (the legacy code included auxiliary variables such as
    CDR in this check); by default the engine variables are used.
    """
    needed = variables if variables is not None else scenario_variables(cfg)
    sub = scenario_df[scenario_df["Variable"].isin(needed)]
    counts = sub.groupby("Region")["Variable"].nunique()
    complete = counts[counts == len(needed)].index.tolist()

    region_map_f = region_map[region_map["Region"].isin(complete)]
    scenario_f = scenario_df[scenario_df["Region"].isin(region_map_f["Region"].unique())]
    dropped = sorted(set(region_map["Region"]) - set(region_map_f["Region"]))
    return scenario_f, region_map_f, dropped
