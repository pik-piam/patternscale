"""Downscaling pipeline: orchestrates the steps for all target variables.

Boundaries:

- inputs must satisfy the data contract (``patternscale.contract``), i.e. all
  source-specific import and cleaning has already happened;
- mask-restricted regional sums (urban, typologies, ...) are computed for
  caller-provided masks; combining, shares, intensities, GHG aggregation and
  region re-aggregation are post-processing and stay outside the package.
"""

from __future__ import annotations

import logging

import dask
import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .contract import validate_inputs
from .core import (
    apply_new_dev_region,
    first_norm_factor,
    intensity_to_da,
    new_dev_mask,
    regional_intensity,
    renorm_factor,
    scale_variable,
)
from .grid import Grid
from .prep import apply_min_proxy
from .results import DownscalingResults, collect_data

logger = logging.getLogger(__name__)


def downscale(
    scenario_df: pd.DataFrame,
    ds: xr.Dataset,
    region_map: pd.DataFrame,
    config: Config,
    masks: dict[str, xr.DataArray] | None = None,
    mask_combinations: dict[str, list[str]] | None = None,
    validate: bool = True,
) -> DownscalingResults:
    """Run the pattern-scaling downscaling for all configured variables.

    Parameters
    ----------
    scenario_df : regional scenario data (contract, long format)
    ds : harmonized grid dataset (contract)
    region_map : regions to downscale (columns Region, Region_number)
    config : run configuration
    masks : optional named 2-D (or time-dependent) masks on the target grid;
        for every mask, regional sums of the downscaled values restricted to
        the mask are reported as ``<variable>|<mask label>``.
    mask_combinations : optional named sums of two mask results, e.g.
        ``{"Urban and Peri-Urban": ["Urban", "Peri-Urban"]}``.
    """
    cfg = config.downscaling
    masks = masks or {}
    mask_combinations = mask_combinations or {}
    for combo, parts in mask_combinations.items():
        unknown = [p for p in parts if p not in masks]
        if len(parts) != 2 or unknown:
            raise ValueError(
                f"mask combination '{combo}' must reference exactly two defined masks, "
                f"got {parts} (unknown: {unknown})"
            )

    if validate:
        validate_inputs(scenario_df, ds, region_map, cfg, strict=True)

    model = config.meta.get("model", "")
    scenario = config.meta.get("scenario", "")
    t0 = cfg.base_year
    factor = cfg.normalization.grid_to_scenario_factor
    skip_first = cfg.normalization.skip_first_norm
    grid = Grid(cfg.grid.resolution_deg)

    # --- pre-corrections --------------------------------------------------
    ds = ds.copy()  # shallow copy; input dataset stays untouched
    if cfg.corrections.min_proxy:
        logger.info("Applying minimum proxy floor to non-empty cells")
        for p in cfg.proxies.values():
            if p.min_value is not None:
                ds[p.grid_var] = apply_min_proxy(
                    ds[p.grid_var], p.min_value, grid,
                    log_diagnostics=config.logging.diagnostics,
                )

    region_number = ds[cfg.region_var]
    nd_cfg = cfg.corrections.new_dev

    tables: list[pd.DataFrame] = []
    grids: dict[str, xr.DataArray] = {}

    for v in cfg.variables:
        logger.info(f"# Calculating {v.short} ({v.name})")

        # gridded base-year pattern [grid units]
        E_t0_g = ds[v.name].sel(time=t0).persist()

        # regional scenario data of target variable and proxy [scenario units]
        E_t_R = scenario_df[scenario_df["Variable"] == v.name].copy()
        p = cfg.proxies[v.proxy]
        P_t_g = ds[p.grid_var].persist()
        P_t_R = scenario_df[scenario_df["Variable"] == p.scenario_var].copy()

        # pattern-scaling kernel
        scal = scale_variable(E_t0_g, P_t_g, E_t_R, P_t_R, region_number, t0)
        E_t = scal.E_t.persist()

        if nd_cfg.enabled:
            intensity_df = regional_intensity(P_t_R, E_t_R, p.intensity_divisor)
            nd_mask = new_dev_mask(
                scal.P_g_rel, scal.P_R_rel_regi, nd_cfg.threshold
            ).persist()

        data: dict[str, list] = {
            "grid_emi": [],
            "regional_sum": [],
            "norm_factor": [],
            "norm_factor_renorm": [],
        }
        mask_data: dict[str, list] = {label: [] for label in masks}

        # --- per-region normalization and corrections ---------------------
        for R in region_map["Region_number"].unique():
            reg = region_map.loc[region_map["Region_number"] == R, "Region"].unique()[0]
            logger.info(f"## Region {reg}")

            mask_R = region_number == R
            E_R = E_t.where(mask_R)

            # regional grid sum in scenario units, zeros -> NaN
            sum_grid_R = E_R.sum(dim=["x", "y"], skipna=True) / factor
            sum_grid_R = sum_grid_R.where(sum_grid_R != 0)

            # regional scenario total
            IAM_R = (
                E_t_R.loc[E_t_R["Region_number"] == R, ["Year", "Value"]]
                .set_index("Year")["Value"]
                .to_xarray()
                .rename({"Year": "time"})
            )

            N_factor = first_norm_factor(IAM_R, sum_grid_R, skip_first)
            N_factor.name = reg
            E_norm_R = E_R * N_factor

            if nd_cfg.enabled:
                IAM_int_da = intensity_to_da(intensity_df, R)
                emi_before = E_norm_R.sum(dim=["x", "y"], skipna=True)

                E_norm_R = apply_new_dev_region(
                    E_norm_R, P_t_g, IAM_int_da, nd_mask, mask_R
                )
                emi_after = E_norm_R.sum(dim=["x", "y"], skipna=True)

                emi_before_c, emi_after_c = dask.compute(emi_before, emi_after)
                logger.info(
                    "Value change by new_dev correction (scenario units): "
                    f"{((emi_after_c - emi_before_c) / factor).values}"
                )

                N_factor_cap = renorm_factor(
                    IAM_R, emi_before_c, emi_after_c, skip_first, factor
                )
                logger.info(f"-> Renormalization factor: {N_factor_cap.values}")
                E_norm_R = E_norm_R * N_factor_cap

                N_factor_cap.name = reg
                data["norm_factor_renorm"].append(N_factor_cap)

            reg_sum = E_norm_R.sum(dim=["x", "y"], skipna=True) / factor
            reg_sum.name = reg

            data["grid_emi"].append(E_norm_R)
            data["regional_sum"].append(reg_sum)
            data["norm_factor"].append(N_factor)

            for label, m in masks.items():
                masked_sum = (E_norm_R * m).sum(dim=["x", "y"], skipna=True) / factor
                masked_sum.name = reg
                mask_data[label].append(masked_sum)

        # --- combine regions ----------------------------------------------
        emi_global = xr.concat(data["grid_emi"], dim="region").sum("region", skipna=True)
        emi_global = emi_global.where(emi_global != 0, np.nan)
        emi_global.name = v.short
        grids[v.short] = emi_global

        # --- regional tables ----------------------------------------------
        emi = collect_data(data["regional_sum"])
        norm_df = collect_data(data["norm_factor"])

        mask_dfs: dict[str, pd.DataFrame] = {
            label: collect_data(mask_data[label]) for label in masks
        }
        combo_dfs: dict[str, pd.DataFrame] = {}
        for combo, (a, b) in mask_combinations.items():
            merged = pd.merge(mask_dfs[a], mask_dfs[b], on=["region", "year"])
            merged["value"] = merged["value_x"] + merged["value_y"]
            merged.drop(columns=["value_x", "value_y"], inplace=True)
            combo_dfs[combo] = merged

        emi["variable"] = f"{v.name}"
        norm_df["variable"] = f"{v.name}|NormFactor"
        for label, mdf in mask_dfs.items():
            mdf["variable"] = f"{v.name}|{label}"
        for combo, cdf in combo_dfs.items():
            cdf["variable"] = f"{v.name}|{combo}"

        frames = [emi, *mask_dfs.values(), *combo_dfs.values(), norm_df]
        if nd_cfg.enabled:
            norm_cap_df = collect_data(data["norm_factor_renorm"])
            norm_cap_df["variable"] = f"{v.name}|NormFactor_EmiCap"
            frames.append(norm_cap_df)

        out = pd.concat(frames, axis=0, ignore_index=True)
        out["scenario"] = scenario
        out["model"] = model
        out = out.pivot(
            index=["model", "scenario", "variable", "region"],
            columns="year",
            values="value",
        ).reset_index()
        tables.append(out)

    table = pd.concat(tables, ignore_index=True)
    return DownscalingResults(table=table, grid=xr.Dataset(grids), meta=dict(config.meta))
