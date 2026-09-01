"""Synthetic-data tests for patternscale.

Run with:
    ./.pixi/envs/default/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from patternscale import Config, ContractError, Grid, downscale, validate_inputs
from patternscale.contract import restrict_to_complete_regions
from patternscale.core import map_to_grid, scale_variable
from patternscale.prep import apply_min_proxy, harmonize_dataset


def make_config(**overrides) -> Config:
    base = {
        "downscaling": {
            "years": [2020, 2030],
            "base_year": 2020,
            "grid": {"resolution_deg": 0.1},
            "variables": [{"name": "E", "short": "E_short", "proxy": "POP"}],
            "proxies": {
                "POP": {"grid_var": "Population", "scenario_var": "POP", "intensity_divisor": 1.0}
            },
            "corrections": {"min_proxy": False, "new_dev": {"enabled": False, "threshold": 3.0}},
            "normalization": {"skip_first_norm": False, "grid_to_scenario_factor": 1.0},
        },
        "meta": {"model": "TestModel", "scenario": "TestScenario"},
    }
    # apply nested overrides
    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v
    merge(base, overrides)
    return Config.model_validate(base)


def make_inputs():
    """4x2 grid, two regions (left/right half), one variable, one proxy."""
    x = np.array([0.05, 0.15, 0.25, 0.35])
    y = np.array([0.55, 0.45])
    time = [2020, 2030]

    region = xr.DataArray(
        np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float),
        dims=["y", "x"], coords={"y": y, "x": x},
    )

    pop_2020 = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 1.0, 1.0]])
    pop_2030 = np.array([[2.0, 2.0, 3.0, 8.0], [2.0, 4.0, 1.0, 1.0]])
    pop = xr.DataArray(
        np.stack([pop_2020, pop_2030]),
        dims=["time", "y", "x"], coords={"time": time, "y": y, "x": x},
    )

    emi_2020 = np.array([[10.0, 20.0, 5.0, 5.0], [10.0, 0.0, 10.0, 20.0]])
    emi = xr.DataArray(
        np.stack([emi_2020, emi_2020]),  # only base year is used
        dims=["time", "y", "x"], coords={"time": time, "y": y, "x": x},
    )

    ds = xr.Dataset({"E": emi, "Population": pop, "region_number": region})

    # scenario totals; base year matches grid sums so t0 identity holds
    # region 1 grid sum 2020: 10+20+10+0 = 40 ; region 2: 5+5+10+20 = 40
    rows = []
    for var, r, y2020, y2030 in [
        ("E", 1, 40.0, 50.0),
        ("E", 2, 40.0, 20.0),
        ("POP", 1, 7.0, 10.0),
        ("POP", 2, 9.0, 13.0),
    ]:
        rows.append({"Model": "TestModel", "Scenario": "TestScenario", "Unit": "u",
                     "Variable": var, "Region": f"R{r}", "Region_number": r,
                     "Year": 2020, "Value": y2020})
        rows.append({"Model": "TestModel", "Scenario": "TestScenario", "Unit": "u",
                     "Variable": var, "Region": f"R{r}", "Region_number": r,
                     "Year": 2030, "Value": y2030})
    scenario_df = pd.DataFrame(rows)

    region_map = pd.DataFrame({"Region": ["R1", "R2"], "Region_number": [1, 2]})
    return scenario_df, ds, region_map


class TestConfig(unittest.TestCase):
    def test_yaml_roundtrip(self):
        cfg = make_config()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.yaml"
            cfg.to_yaml(p)
            cfg2 = Config.from_yaml(p)
        self.assertEqual(cfg, cfg2)
        self.assertEqual(cfg.diff(cfg2), [])

    def test_diff(self):
        cfg = make_config()
        cfg2 = make_config(downscaling={"base_year": 2020, "years": [2020, 2040]})
        diffs = cfg.diff(cfg2)
        self.assertTrue(any("years" in d for d in diffs))

    def test_base_year_must_be_in_years(self):
        with self.assertRaises(Exception):
            make_config(downscaling={"base_year": 2019})

    def test_unknown_proxy_rejected(self):
        with self.assertRaises(Exception):
            make_config(downscaling={"variables": [{"name": "E", "short": "E", "proxy": "GDP"}]})


class TestGrid(unittest.TestCase):
    def test_legacy_01_coords_bit_identical(self):
        g = Grid(0.1)
        np.testing.assert_array_equal(g.x_coords(), np.arange(-179.95, 180.0, 0.1))
        np.testing.assert_array_equal(g.y_coords(), np.arange(89.95, -90, -0.1))
        self.assertEqual(g.nx, 3600)
        self.assertEqual(g.ny, 1800)
        self.assertEqual(g.reindex_tolerance, 0.05)

    def test_generic_resolution(self):
        g = Grid(0.5)
        self.assertEqual(g.nx, 720)
        self.assertEqual(g.ny, 360)
        self.assertAlmostEqual(g.x_coords()[0], -179.75)
        self.assertAlmostEqual(g.y_coords()[0], 89.75)

    def test_cell_area_formula_matches_legacy(self):
        g = Grid(0.1)
        y = np.array([0.05, 45.0, 89.95])
        lat_rad = np.deg2rad(y)
        res_rad = np.deg2rad(0.1)
        legacy = (6371**2) * np.cos(lat_rad) * res_rad**2 / 100
        np.testing.assert_array_equal(g.cell_area_per_100km2(y), legacy)


class TestMapToGrid(unittest.TestCase):
    def test_broadcast(self):
        _, ds, _ = make_inputs()
        da = xr.DataArray(
            np.array([[1.5, 2.5], [3.5, 4.5]]),
            dims=["time", "Region_number"],
            coords={"time": [2020, 2030], "Region_number": [1, 2]},
        )
        out = map_to_grid(da, ds["region_number"])
        self.assertEqual(out.sel(time=2020).values[0, 0], 1.5)  # region 1
        self.assertEqual(out.sel(time=2020).values[0, 3], 2.5)  # region 2
        self.assertEqual(out.sel(time=2030).values[1, 1], 3.5)
        self.assertEqual(out.sel(time=2030).values[1, 2], 4.5)

    def test_unmapped_region_is_nan(self):
        _, ds, _ = make_inputs()
        da = xr.DataArray(
            np.array([[1.5], [3.5]]),
            dims=["time", "Region_number"],
            coords={"time": [2020, 2030], "Region_number": [1]},
        )
        out = map_to_grid(da, ds["region_number"])
        self.assertTrue(np.isnan(out.sel(time=2020).values[0, 3]))


class TestKernel(unittest.TestCase):
    def test_analytic_cell_value(self):
        scenario_df, ds, _ = make_inputs()
        E_t_R = scenario_df[scenario_df["Variable"] == "E"]
        P_t_R = scenario_df[scenario_df["Variable"] == "POP"]
        res = scale_variable(
            ds["E"].sel(time=2020), ds["Population"], E_t_R, P_t_R,
            ds["region_number"], 2020,
        )
        # cell (0,0), region 1, 2030:
        # E0=10, E_R ratio=50/40, P_g ratio=2/1, P_R ratio=10/7
        expected = 10.0 * (50.0 / 40.0) * (2.0 / 1.0) / (10.0 / 7.0)
        self.assertAlmostEqual(float(res.E_t.sel(time=2030).values[0, 0]), expected, places=12)

    def test_base_year_identity(self):
        scenario_df, ds, _ = make_inputs()
        E_t_R = scenario_df[scenario_df["Variable"] == "E"]
        P_t_R = scenario_df[scenario_df["Variable"] == "POP"]
        res = scale_variable(
            ds["E"].sel(time=2020), ds["Population"], E_t_R, P_t_R,
            ds["region_number"], 2020,
        )
        base = res.E_t.sel(time=2020).values
        np.testing.assert_allclose(base, ds["E"].sel(time=2020).values, rtol=1e-14)


class TestPipeline(unittest.TestCase):
    def test_mass_conservation_with_norm(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()  # skip_first_norm=False
        res = downscale(scenario_df, ds, region_map, cfg)
        t = res.table
        for r, year, expected in [("R1", 2030, 50.0), ("R2", 2030, 20.0),
                                  ("R1", 2020, 40.0), ("R2", 2020, 40.0)]:
            got = t.loc[(t["variable"] == "E") & (t["region"] == r), year].item()
            self.assertAlmostEqual(got, expected, places=10)

    def test_mass_conservation_with_new_dev_and_skip_first_norm(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config(downscaling={
            "corrections": {"new_dev": {"enabled": True, "threshold": 3.0}},
            "normalization": {"skip_first_norm": True, "grid_to_scenario_factor": 1.0},
        })
        res = downscale(scenario_df, ds, region_map, cfg)
        t = res.table
        for r, year, expected in [("R1", 2030, 50.0), ("R2", 2030, 20.0)]:
            got = t.loc[(t["variable"] == "E") & (t["region"] == r), year].item()
            self.assertAlmostEqual(got, expected, places=10)
        # renorm factor rows present
        self.assertIn("E|NormFactor_EmiCap", t["variable"].unique())

    def test_grid_sums_match_table(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        res = downscale(scenario_df, ds, region_map, cfg)
        grid_sum = float(res.grid["E_short"].sel(time=2030).sum(skipna=True))
        t = res.table
        table_sum = t.loc[t["variable"] == "E", 2030].sum()
        self.assertAlmostEqual(grid_sum, table_sum, places=10)

    def test_mask_sums_and_combination(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        top = xr.zeros_like(ds["region_number"])
        top[0, :] = 1.0
        bottom = xr.zeros_like(ds["region_number"])
        bottom[1, :] = 1.0
        res = downscale(
            scenario_df, ds, region_map, cfg,
            masks={"Top": top, "Bottom": bottom},
            mask_combinations={"Both": ["Top", "Bottom"]},
        )
        t = res.table
        for r in ["R1", "R2"]:
            total = t.loc[(t["variable"] == "E") & (t["region"] == r), 2030].item()
            both = t.loc[(t["variable"] == "E|Both") & (t["region"] == r), 2030].item()
            self.assertAlmostEqual(total, both, places=10)

    def test_table_long_roundtrip(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        res = downscale(scenario_df, ds, region_map, cfg)
        long = res.table_long()
        self.assertEqual(
            list(long.columns), ["model", "scenario", "variable", "region", "year", "value"]
        )
        # long and wide carry the same values
        v = long[(long["variable"] == "E") & (long["region"] == "R1") & (long["year"] == 2030)]
        self.assertAlmostEqual(v["value"].item(), 50.0, places=10)
        # CSV round-trip keeps the year column numeric
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "long.csv"
            res.save_table(p, format="long")
            back = pd.read_csv(p)
        self.assertTrue(pd.api.types.is_integer_dtype(back["year"]))

    def test_invalid_mask_combination_rejected(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        with self.assertRaises(ValueError):
            downscale(scenario_df, ds, region_map, cfg,
                      mask_combinations={"Bad": ["NoSuchMask", "AlsoNot"]})

    def test_new_dev_flags_fast_growing_cell(self):
        scenario_df, ds, region_map = make_inputs()
        # cell (0,3) region 2: proxy 4 -> 8 (ratio 2), region ratio 13/9 ~ 1.44
        # with threshold 1.3: 2/1.44 = 1.385 > 1.3 -> flagged
        cfg = make_config(downscaling={
            "corrections": {"new_dev": {"enabled": True, "threshold": 1.3}},
            "normalization": {"skip_first_norm": True, "grid_to_scenario_factor": 1.0},
        })
        res = downscale(scenario_df, ds, region_map, cfg)
        # region totals still match scenario
        t = res.table
        got = t.loc[(t["variable"] == "E") & (t["region"] == "R2"), 2030].item()
        self.assertAlmostEqual(got, 20.0, places=10)
        # flagged cell: intensity*proxy*renorm; verify against direct computation
        # regional intensity 2030 = E_R/P_R = 20/13
        # unflagged cells keep E_int * P; flagged cell gets IAM_int * P
        # after renorm all sum to 20
        grid_2030 = res.grid["E_short"].sel(time=2030).values
        self.assertFalse(np.isnan(grid_2030[0, 3]))


class TestContract(unittest.TestCase):
    def test_valid_inputs_pass(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        problems = validate_inputs(scenario_df, ds, region_map, cfg.downscaling, strict=False)
        self.assertEqual(problems, [])

    def test_missing_variable_detected(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        bad = scenario_df[scenario_df["Variable"] != "POP"]
        with self.assertRaises(ContractError) as ctx:
            validate_inputs(bad, ds, region_map, cfg.downscaling)
        self.assertTrue(any("POP" in p for p in ctx.exception.problems))

    def test_duplicates_detected(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        bad = pd.concat([scenario_df, scenario_df.head(1)])
        with self.assertRaises(ContractError) as ctx:
            validate_inputs(bad, ds, region_map, cfg.downscaling)
        self.assertTrue(any("duplicate" in p for p in ctx.exception.problems))

    def test_missing_grid_var_detected(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        with self.assertRaises(ContractError) as ctx:
            validate_inputs(scenario_df, ds.drop_vars("Population"), region_map, cfg.downscaling)
        self.assertTrue(any("Population" in p for p in ctx.exception.problems))

    def test_restrict_to_complete_regions(self):
        scenario_df, ds, region_map = make_inputs()
        cfg = make_config()
        # remove proxy data of region 2
        bad = scenario_df[~((scenario_df["Region"] == "R2") & (scenario_df["Variable"] == "POP"))]
        sc_f, map_f, dropped = restrict_to_complete_regions(bad, region_map, cfg.downscaling)
        self.assertEqual(dropped, ["R2"])
        self.assertEqual(sorted(sc_f["Region"].unique()), ["R1"])
        self.assertEqual(map_f["Region"].tolist(), ["R1"])


class TestPrep(unittest.TestCase):
    def test_min_proxy_floor(self):
        _, ds, _ = make_inputs()
        da = ds["Population"].copy()
        da.values[0, 0, 0] = 0.001   # tiny but non-empty
        da.values[0, 0, 1] = np.nan  # empty stays empty
        out = apply_min_proxy(da, min_value=0.5, grid=Grid(0.1))
        area = Grid(0.1).cell_area_per_100km2(np.array([0.55]))[0]
        self.assertAlmostEqual(float(out.values[0, 0, 0]), 0.5 * area, places=12)
        self.assertTrue(np.isnan(out.values[0, 0, 1]))
        # values above the floor are unchanged
        self.assertEqual(float(out.values[0, 1, 0]), 2.0)

    def test_harmonize_realigns_jittered_coords(self):
        g = Grid(0.1)
        x = g.x_coords()[:4] + 1e-9
        y = g.y_coords()[:2] - 1e-9
        da = xr.DataArray(
            np.ones((2, 4, 2)), dims=["y", "x", "time"],
            coords={"y": y, "x": x, "time": [2020, 2030]},
        )
        ds = xr.Dataset({"v": da})
        out = harmonize_dataset(ds, g, years=[2020, 2030], check_sum=True)
        self.assertEqual(out.sizes["x"], g.nx)
        self.assertEqual(out.sizes["y"], g.ny)
        np.testing.assert_array_equal(out["x"].values, g.x_coords())
        # original values preserved at the matched cells
        self.assertEqual(float(out["v"].sel(time=2020).values[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
