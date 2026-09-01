"""Regular global lat/lon grid specification.

Replaces the hardcoded 0.1-degree coordinate arrays of the legacy code. For
resolution 0.1 the exact legacy literals are used so that coordinates are
bit-identical to previous runs; other resolutions use the generic formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Global, cell-centered regular grid.

    x ascends from -180 to 180, y descends from 90 to -90 (raster order).
    """

    resolution_deg: float

    def x_coords(self) -> np.ndarray:
        if self.resolution_deg == 0.1:
            # exact legacy expression (bit-identical coordinates)
            return np.arange(-179.95, 180.0, 0.1)
        res = self.resolution_deg
        return np.arange(-180.0 + res / 2, 180.0, res)

    def y_coords(self) -> np.ndarray:
        if self.resolution_deg == 0.1:
            # exact legacy expression (bit-identical coordinates)
            return np.arange(89.95, -90, -0.1)
        res = self.resolution_deg
        return np.arange(90.0 - res / 2, -90.0, -res)

    @property
    def nx(self) -> int:
        return len(self.x_coords())

    @property
    def ny(self) -> int:
        return len(self.y_coords())

    @property
    def reindex_tolerance(self) -> float:
        """Tolerance for nearest-neighbour reindexing onto this grid."""
        return self.resolution_deg / 2

    def cell_area_per_100km2(self, y: "np.ndarray | object") -> object:
        """Cell area in units of 100 km^2 as a function of latitude.

        Accepts a numpy array or an xarray coordinate; returns the same type.
        Matches the legacy formula in set_min_proxy.
        """
        lat_rad = np.deg2rad(y)
        res_rad = np.deg2rad(self.resolution_deg)
        return (6371.0**2) * np.cos(lat_rad) * res_rad**2 / 100
