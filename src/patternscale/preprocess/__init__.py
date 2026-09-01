"""Preprocessing: from original data sources to the engine's input objects.

This is a clearly separate step from the downscaling itself: it turns raw
source data (netCDF fluxes, GeoTIFF stacks, country rasters) into the
processed grid datasets that the adapters (``patternscale.adapters``) load
and the engine consumes. It typically runs once per data release, not per
scenario run.

Modules (one per source family):

- ``raster``   : shared low-level readers/writers (GeoTIFF stacks, per-gas
                 netCDF, mode-coarsening, zarr output, base-year averaging)
- ``ceds``     : CEDS CMIP7 monthly emission fluxes -> sectoral annual
                 emission grids [t/yr]
- ``edgar``    : EDGAR annual sectoral emission grids -> variable-mapped
                 emission grids [t/yr]
- ``compass``  : COMPASS 6-arcmin GeoTIFF projections -> population, GDP and
                 urbanization grids
- ``typology`` : city-typology raster -> Type 1-4 mask grids
- ``regions``  : 1-km country raster -> ISO country grid and region-number
                 rasters for a given region mapping ("grid to country")

Output conventions (what downstream code expects; see also
``patternscale.adapters`` and ``patternscale.contract``):

- dims ``y`` (lat, descending), ``x`` (lon, ascending), ``time`` (years, int)
- empty cells NaN (zeros are masked so zarr can skip empty chunks)
- emission variable names follow the IAMC-style ``Emissions|<gas>|<sector>``
  pattern; the ``|`` separator is encoded as ``__`` in zarr stores and
  decoded on load by ``adapters.grids.load_emissions``
- one zarr store per source and resolution; filename conventions are the
  calling project's responsibility

All functions take explicit input/output paths.
"""

from . import ceds, compass, edgar, raster, regions, typology  # noqa: F401
