"""Tests for the preprocessing step (pure functions, synthetic data)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from patternscale.preprocess import ceds, compass, edgar, raster, regions


def make_yearly(values_by_year: dict[int, float], var: str = "v") -> xr.Dataset:
    years = sorted(values_by_year)
    data = np.array([[[values_by_year[y]]] for y in years])  # (year, y, x)
    return xr.Dataset(
        {var: (("year", "y", "x"), data)},
        coords={"year": years, "y": [0.05], "x": [0.05]},
    )


class TestAverageBaseYears(unittest.TestCase):
    def test_exact_window(self):
        ds = make_yearly({y: float(y) for y in range(2003, 2023)})
        out = raster.average_base_years(ds, [2005, 2020], window=2)
        # mean of y-2..y+2 of the identity series is y
        self.assertEqual(float(out["v"].sel(year=2005)), 2005.0)
        self.assertEqual(float(out["v"].sel(year=2020)), 2020.0)

    def test_clipped_window(self):
        # data ends 2022: window for 2020 clips to 2018-2022 (legacy EDGAR)
        ds = make_yearly({y: float(y) for y in range(2018, 2023)})
        out = raster.average_base_years(
            ds, [2020], window=2, out_dim="time", clip_to_available=True
        )
        self.assertEqual(float(out["v"].sel(time=2020)), 2020.0)

    def test_exact_window_raises_on_missing(self):
        ds = make_yearly({y: float(y) for y in range(2019, 2023)})  # 2018 missing
        with self.assertRaises(KeyError):
            raster.average_base_years(ds, [2020], window=2).compute()


class TestEdgarAggregation(unittest.TestCase):
    def _gas_ds(self):
        # two sectors mapping to the same target; disjoint source patterns
        a = np.array([[1.0, np.nan]])
        b = np.array([[np.nan, 2.0]])
        return {
            "CH4": xr.Dataset(
                {"ENE": (("y", "x"), a), "PRO": (("y", "x"), b)},
                coords={"y": [0.05], "x": [0.05, 0.15]},
            )
        }

    def test_nan_safe_collision_sum(self):
        # regression for the historical NaN-propagation bug: plain '+' would
        # lose both cells here
        mapping = {"ENE": "Energy|Supply", "PRO": "Energy|Supply"}
        out = edgar.aggregate_sectors(self._gas_ds(), mapping)
        v = out["Emissions|CH4|Energy|Supply"].values
        np.testing.assert_array_equal(v, [[1.0, 2.0]])

    def test_unmapped_sector_dropped_and_target_filter(self):
        mapping = {"ENE": "Energy|Supply"}  # PRO unmapped
        out = edgar.aggregate_sectors(self._gas_ds(), mapping)
        self.assertEqual(list(out.data_vars), ["Emissions|CH4|Energy|Supply"])
        out2 = edgar.aggregate_sectors(
            self._gas_ds(), {"ENE": "Energy|Supply", "PRO": "Waste"}, target_paths=["Waste"]
        )
        self.assertEqual(list(out2.data_vars), ["Emissions|CH4|Waste"])

    def test_combine_industry(self):
        ds = xr.Dataset(
            {
                "Emissions|CO2|Energy|Demand|Industry": (("y", "x"), np.array([[1.0, np.nan]])),
                "Emissions|CO2|Industrial Processes": (("y", "x"), np.array([[2.0, 3.0]])),
            },
            coords={"y": [0.05], "x": [0.05, 0.15]},
        )
        out = edgar.combine_industry(ds, gases=["CO2"])
        np.testing.assert_array_equal(out["Emissions|CO2|Industry"].values, [[3.0, 3.0]])
        self.assertEqual(list(out.data_vars), ["Emissions|CO2|Industry"])


class TestCedsFormat(unittest.TestCase):
    def test_format_grid(self):
        ds = xr.Dataset(
            {"Emissions|CO2|Waste": (("lat", "lon", "year"), np.array([[[0.0]], [[5.0]]]))},
            coords={"lat": [-0.05, 0.05], "lon": [0.05], "year": [2020]},
        )
        out = ceds.format_grid(ds)
        self.assertEqual(out["Emissions|CO2|Waste"].dims, ("y", "x", "time"))
        self.assertEqual(out["y"].values[0], 0.05)  # descending
        self.assertTrue(np.isnan(out["Emissions|CO2|Waste"].values[1, 0, 0]))  # 0 -> NaN
        self.assertEqual(out["Emissions|CO2|Waste"].dtype, np.float32)


class TestCompass(unittest.TestCase):
    def test_pop_share_masks(self):
        coords = {"y": [0.05], "x": [0.05]}
        city = xr.Dataset({"mask_urb": (("y", "x"), np.array([[30.0]]))}, coords=coords)
        town = xr.Dataset({"mask_peri": (("y", "x"), np.array([[10.0]]))}, coords=coords)
        pop = xr.Dataset({"Population": (("y", "x"), np.array([[100.0]]))}, coords=coords)
        pop = pop.assign_coords(spatial_ref=0)
        city = city.assign_coords(spatial_ref=0)
        town = town.assign_coords(spatial_ref=0)
        out = compass.pop_share_masks(city, town, pop)
        self.assertAlmostEqual(float(out["mask_urb"]), 0.3)
        self.assertAlmostEqual(float(out["mask_peri"]), 0.1)
        self.assertNotIn("spatial_ref", out.coords)


class TestRegions(unittest.TestCase):
    def test_add_region_numbers(self):
        ds = xr.Dataset(
            {"iso_numeric": (("y", "x"), np.array([[276.0, 250.0], [np.nan, 999.0]]))},
            coords={"y": [0.15, 0.05], "x": [0.05, 0.15]},
        )
        mapping = pd.DataFrame({"iso_lookup_numeric": [276, 250], "RegionNumber": [1, 2]})
        out = regions.add_region_numbers(ds, mapping)
        v = out["region_number"].values
        self.assertEqual(v[0, 0], 1.0)
        self.assertEqual(v[0, 1], 2.0)
        self.assertTrue(np.isnan(v[1, 0]))   # NaN stays NaN
        self.assertTrue(np.isnan(v[1, 1]))   # unmapped code -> NaN
        self.assertEqual(out["region_number"].dtype, np.float32)


class TestRasterHelpers(unittest.TestCase):
    def test_mask_zeros_and_pipe_encoding(self):
        ds = xr.Dataset(
            {"Emissions|CO2|Waste": (("y", "x"), np.array([[0.0, 4.0]]))},
            coords={"y": [0.05], "x": [0.05, 0.15]},
        )
        masked = raster.mask_zeros(ds)
        self.assertTrue(np.isnan(masked["Emissions|CO2|Waste"].values[0, 0]))
        enc = raster.encode_pipe_names(ds)
        self.assertEqual(list(enc.data_vars), ["Emissions__CO2__Waste"])


if __name__ == "__main__":
    unittest.main()
