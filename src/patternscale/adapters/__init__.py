"""Format-level adapters: readers and transforms for specific data formats.

Adapters bridge external data formats and the engine's data contract
(authoritative definition and validators: ``patternscale.contract``). They
take explicit file paths or already-opened objects — directory layouts and
filename conventions belong to the calling project, not here.

Adding a new data source means producing these structures, then validating
with ``patternscale.validate_inputs``:

1. Scenario data -> long ``pandas.DataFrame`` with columns
   - ``Variable`` (str), ``Region`` (str), ``Region_number`` (int),
     ``Year`` (int), ``Value`` (float); optional ``Model``, ``Scenario``,
     ``Unit`` are passed through
   - one row per (Variable, Region, Year), no duplicates
   - must cover every ``VariableSpec.name`` and ``ProxySpec.scenario_var``
     for the base year and all downscaling years
   - units must be consistent with ``ProxySpec.intensity_divisor`` and
     ``NormalizationConfig.grid_to_scenario_factor``; unit conversions
     (e.g. the legacy N2O kt -> Mt workaround) happen in the adapter

2. Grid data -> ``xarray.Dataset``/``DataArray`` with dims
   - ``y`` (latitude, descending), ``x`` (longitude, ascending), and for
     time-dependent fields ``time`` (years as integers; a ``year`` dim is
     renamed by ``prep.harmonize_dataset``)
   - data variables named exactly as referenced in the config
     (``VariableSpec.name`` for targets, ``ProxySpec.grid_var`` for proxies)
   - empty cells are NaN, not 0
   - coordinates need not be exact: ``prep.harmonize_dataset`` reindexes
     onto the target ``Grid`` (nearest neighbour, half-cell tolerance) and
     verifies no data is lost

3. Region raster -> 2-D field (``DownscalingConfig.region_var``) holding the
   integer region number per cell (NaN outside), plus a region map
   ``DataFrame`` with columns ``Region`` and ``Region_number`` linking the
   raster to the scenario data.
"""
