"""Adapter for ScenarioMIP country-level scenario files (SMIP_Countries)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUX_VARIABLES = {"CO2_CDR": "Emissions|CO2|CDR", "Population": "Population", "GDP": "GDP|PPP"}


def load_scenario(
    path: str | Path,
    target_variables: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load one SMIP country scenario file (long-format IAMC CSV).

    ``target_variables`` maps short names to IAMC variable names. Returns
    (full frame, frame filtered to target + auxiliary variables, detected
    model name).
    """
    df_all = pd.read_csv(path)
    model = df_all["Model"][0]

    variables = {**target_variables, **AUX_VARIABLES}
    df = df_all[df_all["Variable"].isin(variables.values())]
    return df_all, df, model


def clean_scenario(
    df: pd.DataFrame,
    region_map: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Attach region numbers, filter years, convert N2O from kt to Mt.

    The N2O conversion keeps computations in comparable magnitudes and is
    reverted in post-processing (legacy behaviour; the Unit column is not
    updated).
    """
    df = df.copy()
    region_to_num = dict(zip(region_map["Region"], region_map["Region_number"]))
    df.loc[:, "Region_number"] = df["Region"].map(region_to_num)
    df = df[df["Year"].isin(years)]

    df.loc[df["Variable"].str.contains("N2O"), "Value"] = (
        df.loc[df["Variable"].str.contains("N2O"), "Value"] / 1000
    )

    df = df[~df["Region_number"].isna()]
    df["Region_number"] = df["Region_number"].astype(int)
    return df
