# table_a2.py
from pathlib import Path
import numpy as np
import pandas as pd

# Repo base (EDS)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folder outside repo
DATA_DIR = BASE_DIR.parent / "Replication Project" / "Data"

INPUT_FILE = DATA_DIR / "Final" / "acs_ssc_final_v2.pkl"
OUTPUT_DIR = BASE_DIR / "Data" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "table_A2.csv"
OUTPUT_XLSX = OUTPUT_DIR / "table_A2.xlsx"


def format_cell(mean_val, sd_val, decimals=3):
    """Return a paper-style cell with mean on first line and sd in parentheses below."""
    if pd.isna(mean_val):
        return ""
    if pd.isna(sd_val):
        return f"{mean_val:,.{decimals}f}"
    return f"{mean_val:,.{decimals}f}\n({sd_val:,.{decimals}f})"


def build_table_a2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Table A2:
    Summary statistics for same-sex married and cohabiting couples.
    Assumes df is already restricted to the paper sample in build_data.py.
    """
    df = df.copy()

    # Additional variables needed for the table
    df["female"] = 1 - df["r_male"]

    # Conditional number of dependent children:
    # only defined for couples with any dependent children
    df["c_children_cond"] = np.where(
        df["c_anychildren"] == 1,
        df["c_children"],
        np.nan
    )

    # Variables and labels in table order
    variables = [
        ("Male", "r_male"),
        ("Female", "female"),
        ("Partners are the same race", "c_racecomp"),
        ("Age of older partner", "c_agemax"),
        ("Age of younger partner", "c_agemin"),
        ("Age difference between partners", "c_agediff"),
        ("Education of more educated partner", "c_edumax"),
        ("Education of less educated partner", "c_edumin"),
        ("Education difference between partners", "c_edudiff"),
        ("Any dependent children", "c_anychildren"),
        ("Conditional number of dependent children", "c_children_cond"),
        ("Both partners work", "c_dualearner"),
        ("Only 1 partner works", "c_singleearner"),
        ("Neither partner works", "c_noearner"),
    ]

    # Check group values
    group_values = sorted(df["sscouple_mar"].dropna().unique().tolist())
    if set(group_values) != {False, True} and set(group_values) != {0, 1}:
        raise ValueError(
            f"Unexpected values in sscouple_mar: {group_values}. "
            "Expected boolean or 0/1."
        )

    grouped = df.groupby("sscouple_mar", dropna=False)

    rows = []
    for label, var in variables:
        stats = grouped[var].agg(["mean", "std"])

        married_key = True if True in stats.index else 1
        cohab_key = False if False in stats.index else 0

        married_mean = stats.loc[married_key, "mean"]
        married_sd = stats.loc[married_key, "std"]
        cohab_mean = stats.loc[cohab_key, "mean"]
        cohab_sd = stats.loc[cohab_key, "std"]

        rows.append({
            "Variable": label,
            "Married couples": format_cell(married_mean, married_sd),
            "Cohabiting couples": format_cell(cohab_mean, cohab_sd),
            "married_mean": married_mean,
            "married_sd": married_sd,
            "cohab_mean": cohab_mean,
            "cohab_sd": cohab_sd,
        })

    # Add observations row
    counts = grouped.size()
    married_key = True if True in counts.index else 1
    cohab_key = False if False in counts.index else 0

    rows.append({
        "Variable": "Observations",
        "Married couples": f"{int(counts.loc[married_key]):,}",
        "Cohabiting couples": f"{int(counts.loc[cohab_key]):,}",
        "married_mean": counts.loc[married_key],
        "married_sd": np.nan,
        "cohab_mean": counts.loc[cohab_key],
        "cohab_sd": np.nan,
    })

    table = pd.DataFrame(rows)
    return table


def main():
    print(f"Loading cleaned dataset from:\n{INPUT_FILE}")
    print(f"Input exists: {INPUT_FILE.exists()}")

    df = pd.read_pickle(INPUT_FILE)

    print("\nDataset loaded.")
    print(f"Shape: {df.shape}")

    print("\nChecking group counts...")
    print(df["sscouple_mar"].value_counts(dropna=False))

    print("\n--- DEBUG: CHILDREN ---")
    print("Share with children:", df["c_anychildren"].mean())
    print("Mean children:", df["c_children"].mean())
    print("\nChildren by group:")
    print(df.groupby("sscouple_mar")[["c_anychildren", "c_children"]].mean())

    table_a2 = build_table_a2(df)

    # Paper-style display version
    display_table = table_a2[["Variable", "Married couples", "Cohabiting couples"]].copy()

    print("\nTable A2:")
    print(display_table.to_string(index=False))

    # Save formatted table
    display_table.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV to:   {OUTPUT_CSV}")

    try:
        display_table.to_excel(OUTPUT_XLSX, index=False)
        print(f"Saved Excel to: {OUTPUT_XLSX}")
    except ModuleNotFoundError:
        print("\nExcel file not saved because openpyxl is not installed.")
        print("Run: pip install openpyxl")


if __name__ == "__main__":
    main()