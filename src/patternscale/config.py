"""Typed run configuration for patternscale.

The configuration is split into:

- ``downscaling``: everything that defines the scientific method
  (years, grid, target variables, proxies, corrections, normalization).
- ``output``: what to save and where.
- ``logging``: log level and optional log file.
- ``meta``: free-form labels (model, scenario, ...) that are attached to
  outputs but never interpreted by the engine.

All models are pydantic v2 models; unknown keys are rejected so that typos in
YAML files fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridConfig(StrictModel):
    """Definition of the regular lat/lon target grid.

    The grid is global, cell-centered, with ``x`` ascending from -180 to 180
    and ``y`` descending from 90 to -90.
    """

    resolution_deg: float = 0.1


class ProxySpec(StrictModel):
    """One proxy dataset (e.g. population or GDP).

    intensity_divisor converts the ratio of scenario value over scenario proxy
    (e.g. Mt / billion USD) to the unit of "grid value per grid-proxy unit"
    (e.g. t / USD): intensity = (E_R / P_R) / intensity_divisor. It is only
    used by intensity-based corrections (new_dev); the pattern-scaling kernel
    itself is unit-agnostic. A divisor (not a factor) is used to remain
    bit-identical with the legacy computation (x/1000 != x*0.001 in floating
    point).

    min_value is the optional floor applied to non-empty grid cells by the
    min_proxy correction, in proxy units per 100 km^2 (cell-area scaled).
    """

    grid_var: str
    scenario_var: str
    intensity_divisor: float = 1.0
    min_value: Optional[float] = None


class VariableSpec(StrictModel):
    """One target variable to downscale.

    ``name`` must match both the data variable name in the grid dataset and
    the ``Variable`` entry in the scenario data. ``short`` is the key used for
    the variable in the gridded output. ``proxy`` refers to a key of
    ``DownscalingConfig.proxies``.
    """

    name: str
    short: str
    proxy: str


class NewDevConfig(StrictModel):
    """Correction for newly developed cells.

    Cells whose relative proxy growth exceeds the regional relative proxy
    growth by more than ``threshold`` are set to the regional mean emission
    intensity, followed by a renormalization to the regional total.
    """

    enabled: bool = False
    threshold: float = 3.0


class CorrectionsConfig(StrictModel):
    min_proxy: bool = False
    new_dev: NewDevConfig = Field(default_factory=NewDevConfig)


class NormalizationConfig(StrictModel):
    """Normalization of grid sums to regional scenario totals.

    skip_first_norm=True skips the normalization directly after the scaling
    kernel; the renormalization inside the new_dev correction then targets the
    scenario total directly (matching the legacy behaviour).
    """

    skip_first_norm: bool = True
    # factor between grid value units and scenario value units:
    # scenario_value = grid_sum / grid_to_scenario_factor
    # (legacy: grid in t, scenario in Mt -> 1e6)
    grid_to_scenario_factor: float = 1.0e6


class DownscalingConfig(StrictModel):
    years: list[int]
    base_year: int
    grid: GridConfig = Field(default_factory=GridConfig)
    region_var: str = "region_number"
    variables: list[VariableSpec]
    proxies: dict[str, ProxySpec]
    corrections: CorrectionsConfig = Field(default_factory=CorrectionsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> "DownscalingConfig":
        if self.base_year not in self.years:
            raise ValueError(
                f"base_year {self.base_year} must be included in years {self.years}"
            )
        shorts = [v.short for v in self.variables]
        if len(set(shorts)) != len(shorts):
            raise ValueError("VariableSpec.short values must be unique")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("VariableSpec.name values must be unique")
        missing = {v.proxy for v in self.variables} - set(self.proxies)
        if missing:
            raise ValueError(f"variables reference undefined proxies: {sorted(missing)}")
        return self


class OutputConfig(StrictModel):
    dir: Optional[str] = None
    save_grid: bool = True
    save_tables: bool = True
    save_config: bool = True


class LoggingConfig(StrictModel):
    level: str = "INFO"
    file: Optional[str] = None
    # log expensive diagnostics (forces extra dask computations)
    diagnostics: bool = False


class Config(StrictModel):
    downscaling: DownscalingConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    meta: dict[str, Any] = Field(default_factory=dict)

    # --- YAML round-trip -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, sort_keys=False)

    def diff(self, other: "Config", ignore: set[str] = frozenset({"meta"})) -> list[str]:
        """Return a list of human-readable differences between two configs."""
        return _diff_dict(self.model_dump(), other.model_dump(), ignore_top=ignore)


def _diff_dict(d1: dict, d2: dict, path: str = "", ignore_top: set[str] = frozenset()) -> list[str]:
    out: list[str] = []
    for k in sorted(d1.keys() | d2.keys(), key=str):
        if not path and k in ignore_top:
            continue
        p = f"{path}.{k}" if path else str(k)
        if k not in d1:
            out.append(f"+ {p}: {d2[k]}")
        elif k not in d2:
            out.append(f"- {p}: {d1[k]}")
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            out.extend(_diff_dict(d1[k], d2[k], p))
        elif d1[k] != d2[k]:
            out.append(f"~ {p}: {d1[k]} -> {d2[k]}")
    return out
