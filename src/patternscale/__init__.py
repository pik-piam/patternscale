"""patternscale: proxy-based pattern scaling of regional scenario data to grids.

Intended usage is a single import, with everything reachable as ``ps.<name>``::

    import patternscale as ps

    config = ps.Config.from_yaml("config.yaml")
    ds = ps.harmonize_dataset(ds_raw, ps.Grid(0.1), years=config.downscaling.years)
    results = ps.downscale(scenario_df, ds, region_map, config)

Format-level adapters are exposed as submodules, e.g. ``ps.smip.load_scenario``
and ``ps.grids.load_region_raster``; the required structure of adapter outputs
is documented in ``patternscale.adapters``. The individual computation steps
live in ``patternscale.core`` and can be imported from there for debugging.
"""

from . import preprocess
from .adapters import grids, remind, smip
from .aggregate import regional_mask_sums
from .config import Config, DownscalingConfig, ProxySpec, VariableSpec
from .contract import ContractError, restrict_to_complete_regions, validate_inputs
from .grid import Grid
from .logutils import setup_logging
from .pipeline import downscale
from .prep import apply_min_proxy, harmonize_dataset
from .results import DownscalingResults, collect_data

__all__ = [
    # configuration
    "Config",
    "DownscalingConfig",
    "ProxySpec",
    "VariableSpec",
    # validation
    "ContractError",
    "validate_inputs",
    "restrict_to_complete_regions",
    # preparation
    "Grid",
    "harmonize_dataset",
    "apply_min_proxy",
    # downscaling
    "downscale",
    # results and aggregation
    "DownscalingResults",
    "collect_data",
    "regional_mask_sums",
    # utilities
    "setup_logging",
    # format-level adapters (submodules)
    "grids",
    "smip",
    "remind",
    # preprocessing (separate step: original sources -> processed inputs)
    "preprocess",
]

__version__ = "0.1.0"
