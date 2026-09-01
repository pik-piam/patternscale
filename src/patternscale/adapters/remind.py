"""Adapter for wide IAMC scenario CSVs as used for REMIND runs.

Currently unused by the SMIP workflow but kept for future REMIND runs. Covers
reading/filtering the wide IAMC CSV (``read_iam_data``), the generic cleaning
step (``clean_scenario``) and the REMIND-specific processing
(``process_regional_totals``, ``fix_region_map``).

Interface (see ``patternscale.contract`` for the authoritative definition):
``read_iam_data`` + ``clean_scenario`` must together produce the long
scenario DataFrame the engine consumes — columns Model, Scenario, Region,
Variable, Unit, Year (int), Value, plus Region_number (int) after cleaning,
covering every target variable and proxy for the base year and all target
years, without duplicates on (Variable, Region, Year).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUX_VARIABLES = {"CO2_CDR": "Emissions|CO2|CDR", "Population": "Population", "GDP": "GDP|PPP"}


def read_iam_data(
    filepath: str | Path,
    scenario_name: str,
    model: str,
    model_short: str,
    modelname_in_region: bool,
    target_variables: dict[str, str],
    years: list[int],
    region_world: str = "World",
) -> pd.DataFrame:
    """Read and filter a wide IAMC CSV; returns a long DataFrame.

    Applies the legacy variable-name adjustments (gross supply/industry/
    processes).
    """
    df_IAM = pd.read_csv(filepath)

    variables = {**target_variables, **AUX_VARIABLES}
    variables["CO2_Supply"] = "Gross " + variables["CO2_Supply"]
    variables["CO2_Energy_Industry"] = "Gross Emissions|CO2|Energy|Demand|Industry"
    variables["CO2_Processes"] = "Gross Emissions|CO2|Industrial Processes"

    # only continue working with necessary data
    df_IAM = df_IAM[df_IAM["Region"] != region_world]
    if modelname_in_region:
        df_IAM = df_IAM[df_IAM.Region.str.contains(model_short)]
    if len(df_IAM) == 0:
        raise ValueError("No data found for variables.")

    df_IAM = df_IAM[df_IAM["Model"] == model]
    if len(df_IAM) == 0:
        raise ValueError("No data found for model.")

    df_IAM = df_IAM[df_IAM["Scenario"] == scenario_name]
    if len(df_IAM) == 0:
        raise ValueError("No data found for scenario name.")

    df_IAM = df_IAM[df_IAM["Variable"].isin(variables.values())]
    missing_vars = [v for v in variables.values() if v not in df_IAM["Variable"].unique()]
    if missing_vars:
        print(f"Warning: The following variables are missing in the IAM projection data: {missing_vars}")

    # go long, filter years
    df_IAM = df_IAM.melt(
        id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
        var_name="Year",
        value_name="Value",
    )
    df_IAM["Year"] = df_IAM["Year"].astype(int)
    df_IAM = df_IAM[df_IAM["Year"].isin(years)]

    return df_IAM


def clean_scenario(
    df: pd.DataFrame,
    region_map: pd.DataFrame,
    years: list[int],
    model: str | None = None,
    modelname_in_region: bool = False,
) -> pd.DataFrame:
    """Region-name cleaning, region numbers, year filter, N2O kt -> Mt."""
    df = df.copy()
    if modelname_in_region:
        df["Region"] = df["Region"].str.replace(f"{model}|" or ",", "")

    region_to_num = dict(zip(region_map["Region"], region_map["Region_number"]))
    df.loc[:, "Region_number"] = df["Region"].map(region_to_num)
    df = df[df["Year"].isin(years)]

    df.loc[df["Variable"].str.contains("N2O"), "Value"] = (
        df.loc[df["Variable"].str.contains("N2O"), "Value"] / 1000
    )

    df = df[~df["Region_number"].isna()]
    df["Region_number"] = df["Region_number"].astype(int)
    return df


def process_regional_totals(df: pd.DataFrame) -> pd.DataFrame:
    """CO2 total excluding bunkers, domestic aviation/shipping and AFOLU."""
    new_total_var = "Gross Emissions|CO2|Excl. shipping, aviation, AFOLU"
    emi_vars = [
        "Gross Emissions|CO2",
        "Emissions|CO2|Energy|Demand|Bunkers|International Aviation",
        "Emissions|CO2|Energy|Demand|Bunkers|International Shipping",
        "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
        "Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
        "Emissions|CO2|AFOLU",
    ]

    missing_vars = [var for var in emi_vars if var not in df["Variable"].unique()]
    if missing_vars:
        print(f"Warning: The following variables are missing in the IAM projection data: {missing_vars}")

    df_emi = df[df["Variable"].isin(emi_vars)]
    unit_CO2 = df_emi[df_emi["Variable"] == "Gross Emissions|CO2"]["Unit"].unique()[0]

    # zero out international bunkers on regional level
    # (note: 'Shiping' typo preserved from the legacy code — the mask never
    # matched, so international shipping was in fact not zeroed)
    mask_CO2_int_shipping = (df_emi["Region"] != "World") & (
        df_emi["Variable"] == "Emissions|CO2|Energy|Demand|Bunkers|International Shiping"
    )
    df_emi.loc[mask_CO2_int_shipping, "Value"] = 0

    mask_CO2_int_aviation = (df_emi["Region"] != "World") & (
        df_emi["Variable"] == "Emissions|CO2|Energy|Demand|Bunkers|International Aviation"
    )
    df_emi.loc[mask_CO2_int_aviation, "Value"] = 0

    index_cols = ["Model", "Scenario", "Region"] + (
        ["Region_number"] if "Region_number" in df_emi.columns else []
    ) + ["Year"]
    df_emi = df_emi.pivot(index=index_cols, columns="Variable", values="Value").reset_index()
    df_emi[new_total_var] = (
        df_emi["Gross Emissions|CO2"]
        - df_emi["Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation"]
        - df_emi["Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping"]
        - df_emi["Emissions|CO2|AFOLU"]
    )
    df_emi = df_emi.melt(
        id_vars=index_cols, value_vars=[new_total_var], var_name="Variable", value_name="Value"
    )
    df_emi["Year"] = df_emi["Year"].astype(int)
    df_emi["Unit"] = unit_CO2

    return df_emi


def fix_region_map(region_map: pd.DataFrame) -> pd.DataFrame:
    """Harmonize REMIND region names with the number mapping."""
    region_map = region_map.copy()
    region_map.loc[
        region_map["Region"] == "Canada Australia New Zealand", "Region"
    ] = "Canada, Australia, New Zealand"
    return region_map
