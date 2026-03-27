# build_data.py
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "Data" / "Intermediate" / "acs_ssc_reduced_v2.pkl"
OUTPUT_FILE = BASE_DIR / "Data" / "Final" / "acs_ssc_final_v2.pkl"


def map_years_education(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[s <= 12] = 0
    out.loc[s == 14] = 1
    out.loc[s == 15] = 2
    out.loc[s == 16] = 3
    out.loc[s == 17] = 4
    out.loc[s == 22] = 5
    out.loc[s == 23] = 6
    out.loc[s == 25] = 7
    out.loc[s == 26] = 8
    out.loc[s == 30] = 9
    out.loc[s == 40] = 10
    out.loc[s == 50] = 11
    out.loc[s.between(61, 65, inclusive="both")] = 12
    out.loc[s == 71] = 13
    out.loc[s == 81] = 14
    out.loc[s == 101] = 16
    out.loc[s >= 114] = 18
    return out


def map_education_group(yrsedu: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=yrsedu.index, dtype="float64")
    out.loc[yrsedu < 12] = 1
    out.loc[yrsedu == 12] = 2
    out.loc[(yrsedu > 12) & (yrsedu < 16)] = 3
    out.loc[yrsedu >= 16] = 4
    return out


def map_race(race: pd.Series, hispan: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=race.index, dtype="float64")
    out.loc[(race == 2) & (hispan == 0)] = 1
    out.loc[race.between(4, 6) & (hispan == 0)] = 2
    out.loc[((race == 3) | (race >= 7)) & (hispan == 0)] = 3
    out.loc[(race == 1) & (hispan == 0)] = 4
    out.loc[hispan > 0] = 5
    return out


def map_age_group(age: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=age.index, dtype="float64")
    for i in range(20, 105, 5):
        out.loc[(age <= i) & (age > i - 5)] = i
    return out


def map_occ_broad(occ: pd.Series) -> pd.Series:
    out = pd.Series(0, index=occ.index, dtype="int64")
    out.loc[((occ >= 10) & (occ <= 420)) | (occ == 4465) | (occ == 430)] = 1
    out.loc[((occ >= 500) & (occ <= 740) & (~occ.isin([726, 725, 630]))) | (occ == 425)] = 2
    out.loc[(occ >= 800) & (occ <= 950)] = 3
    out.loc[(occ >= 1007) & (occ <= 1240)] = 4
    out.loc[(occ >= 1300) & (occ <= 1560)] = 5
    out.loc[(occ >= 1600) & (occ <= 1965)] = 6
    out.loc[(occ >= 2000) & (occ <= 2060)] = 7
    out.loc[(occ >= 2100) & (occ <= 2160)] = 8
    out.loc[(occ >= 2200) & (occ <= 2550)] = 9
    out.loc[((occ >= 2600) & (occ <= 2920)) | (occ == 725)] = 10
    out.loc[(occ >= 3000) & (occ <= 3540)] = 11
    out.loc[(occ >= 3600) & (occ <= 3655)] = 12
    out.loc[(occ >= 3700) & (occ <= 3955)] = 13
    out.loc[(occ >= 4000) & (occ <= 4150)] = 14
    out.loc[(occ >= 4200) & (occ <= 4250)] = 15
    out.loc[((occ >= 4300) & (occ <= 4650) & (occ != 4465)) | occ.isin([9050, 9415])] = 16
    out.loc[((occ >= 4700) & (occ <= 4965)) | (occ == 726)] = 17
    out.loc[(occ >= 5000) & (occ <= 5940)] = 18
    out.loc[((occ >= 6005) & (occ <= 6130)) | (occ == 630)] = 19
    out.loc[(occ >= 6200) & (occ <= 6765)] = 20
    out.loc[(occ >= 6800) & (occ <= 6940)] = 21
    out.loc[(occ >= 7000) & (occ <= 7630)] = 22
    out.loc[(occ >= 7700) & (occ <= 8965)] = 23
    out.loc[((occ >= 9000) & (occ <= 9750) & (~occ.isin([9050, 9415])))] = 24
    out.loc[(occ >= 9800) & (occ <= 9920)] = 25
    return out


def map_deg_broad(degfield: pd.Series) -> pd.Series:
    out = pd.Series(0, index=degfield.index, dtype="int64")
    out.loc[degfield.isin([11, 13])] = 1
    out.loc[degfield == 14] = 2
    out.loc[degfield.isin([15, 29, 52, 54, 55])] = 3
    out.loc[degfield.isin([19, 20])] = 4
    out.loc[degfield == 21] = 5
    out.loc[degfield.isin([22, 35, 53, 56, 57, 59])] = 6
    out.loc[degfield == 23] = 7
    out.loc[degfield.isin([24, 25, 38])] = 8
    out.loc[degfield.isin([26, 33, 34, 48, 49, 60, 64])] = 9
    out.loc[degfield == 32] = 10
    out.loc[degfield.isin([36, 37, 50, 51])] = 11
    out.loc[degfield == 61] = 12
    out.loc[degfield == 62] = 13
    out.loc[degfield.isin([0, 40, 41])] = 14
    return out


def add_policy_variables(df: pd.DataFrame) -> pd.DataFrame:
    pre_windsor_states = {6, 9, 11, 19, 23, 24, 25, 33, 36, 50, 53}
    pre_2012_states = {9, 11, 19, 23, 25, 33, 36, 50, 53}

    df["staterecog"] = df["statefip"].isin(pre_windsor_states).astype(int)
    df["staterecog2"] = df["statefip"].isin(pre_2012_states).astype(int)

    ssm_start = {
        2: 2014, 4: 2014, 6: 2013, 8: 2014, 9: 2008, 10: 2013, 11: 2010,
        12: 2015, 15: 2013, 16: 2014, 17: 2014, 18: 2014, 19: 2009,
        23: 2012, 24: 2013, 25: 2004, 27: 2013, 30: 2014, 32: 2014,
        33: 2010, 34: 2013, 35: 2013, 36: 2011, 37: 2014, 40: 2014,
        41: 2014, 42: 2014, 44: 2013, 45: 2014, 49: 2014, 50: 2009,
        51: 2014, 53: 2012, 54: 2014, 55: 2014, 56: 2014
    }

    df["staterecog_policy"] = (
        df["year"] >= df["statefip"].map(ssm_start).fillna(2015)
    ).astype(int)

    df["staterecog_time"] = df["year"] - df["statefip"].map(ssm_start).fillna(2015)

    medicaid_start = {
        2: 2015, 4: 2014, 5: 2014, 6: 2014, 8: 2014, 9: 2014, 10: 2014,
        11: 2014, 15: 2014, 17: 2014, 18: 2015, 19: 2014, 21: 2014,
        22: 2016, 24: 2014, 25: 2014, 26: 2014, 27: 2014, 30: 2016,
        32: 2014, 33: 2014, 34: 2014, 35: 2014, 36: 2014, 38: 2014,
        39: 2014, 41: 2014, 42: 2015, 44: 2014, 50: 2014, 53: 2014,
        54: 2014
    }

    df["medicaid_exp"] = (
        df["year"] >= df["statefip"].map(medicaid_start).fillna(9999)
    ).astype(int)

    df["preW"] = (df["year"] <= 2012).astype(int)
    df["postWpreO"] = ((df["year"] >= 2013) & (df["year"] <= 2014)).astype(int)
    df["postO"] = (df["year"] >= 2015).astype(int)

    return df


def add_spouse_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    spouse_cols = [
        "year", "serial", "pernum", "sex", "qrelate",
        "age", "r_related", "r_relate",
        "r_race", "r_yrsedu", "r_edugroup",
        "r_wkswrkd", "r_hrswrkd", "r_loghrswrkd",
        "r_male", "r_female", "r_logincearn", "r_posincearn",
        "r_incearn", "r_incnonlabor", "r_inctot", "r_incwage", "r_incbus00",
        "r_incinvst", "r_incretir", "r_incss", "r_incwelfr", "r_incsupp",
        "r_incother", "r_lfp", "r_occbroad", "r_degbroad", "r_agegroup"
    ]

    sp = df[spouse_cols].copy().rename(columns={
        "pernum": "sp_pernum",
        "sex": "sp_sex",
        "qrelate": "sp_qrelate",
        "age": "sp_age",
        "r_related": "sp_related",
        "r_relate": "sp_relate",
        "r_race": "sp_race",
        "r_yrsedu": "sp_yrsedu",
        "r_edugroup": "sp_edugroup",
        "r_wkswrkd": "sp_wkswrkd",
        "r_hrswrkd": "sp_hrswrkd",
        "r_loghrswrkd": "sp_loghrswrkd",
        "r_male": "sp_male",
        "r_female": "sp_female",
        "r_logincearn": "sp_logincearn",
        "r_posincearn": "sp_posincearn",
        "r_incearn": "sp_incearn",
        "r_incnonlabor": "sp_incnonlabor",
        "r_inctot": "sp_inctot",
        "r_incwage": "sp_incwage",
        "r_incbus00": "sp_incbus00",
        "r_incinvst": "sp_incinvst",
        "r_incretir": "sp_incretir",
        "r_incss": "sp_incss",
        "r_incwelfr": "sp_incwelfr",
        "r_incsupp": "sp_incsupp",
        "r_incother": "sp_incother",
        "r_lfp": "sp_lfp",
        "r_occbroad": "sp_occbroad",
        "r_degbroad": "sp_degbroad",
        "r_agegroup": "sp_agegroup"
    })

    return df.merge(
        sp,
        left_on=["year", "serial", "sploc"],
        right_on=["year", "serial", "sp_pernum"],
        how="left",
        validate="many_to_one"
    )


def main():
    print("Loading reduced ACS same-sex-couple microdata...")
    df = pd.read_pickle(INPUT_FILE)
    print("Input shape:", df.shape)

    rename_map = {
        "inctot": "r_inctot",
        "incwage": "r_incwage",
        "incbus00": "r_incbus00",
        "incinvst": "r_incinvst",
        "incretir": "r_incretir",
        "incss": "r_incss",
        "incwelfr": "r_incwelfr",
        "incsupp": "r_incsupp",
        "incother": "r_incother",
        "incearn": "r_incearn",
        "relate": "r_relate",
        "related": "r_related",
    }
    df = df.rename(columns=rename_map)

    # Clean IPUMS income missings
    na_values = {
        "ftotinc": 9999999,
        "r_inctot": 9999999,
        "r_incwage": 999999,
        "r_incbus00": 999999,
        "r_incinvst": 999999,
        "r_incretir": 999999,
        "r_incss": 99999,
        "r_incwelfr": 99999,
        "r_incsupp": 99999,
        "r_incother": 99999,
    }
    for col, bad in na_values.items():
        if col in df.columns:
            df.loc[df[col] == bad, col] = 0

    df.loc[df["r_incbus00"] == 1, "r_incbus00"] = 0
    df.loc[df["r_inctot"] == 1, "r_inctot"] = 0

    # Respondent characteristics
    df["r_logincearn"] = np.log(df["r_incearn"] + 1)
    df["r_posincearn"] = (df["r_incearn"] > 0).astype(int)
    df["r_incnonlabor"] = df["r_inctot"] - df["r_incearn"] - df["r_incss"]
    df["r_yrsedu"] = map_years_education(df["educd"])
    df["r_edugroup"] = map_education_group(df["r_yrsedu"])
    df["r_race"] = map_race(df["race"], df["hispan"])
    df["r_black"] = (df["r_race"] == 1).astype(int)
    df["r_asian"] = (df["r_race"] == 2).astype(int)
    df["r_other"] = (df["r_race"] == 3).astype(int)
    df["r_white"] = (df["r_race"] == 4).astype(int)
    df["r_hispanic"] = (df["r_race"] == 5).astype(int)
    df["r_agegroup"] = map_age_group(df["age"])
    df["r_occbroad"] = map_occ_broad(df["occ"])
    df["r_degbroad"] = map_deg_broad(df["degfield"])
    df["r_male"] = (df["sex"] == 1).astype(int)
    df["r_female"] = (df["sex"] == 2).astype(int)

    df["r_wkswrkd"] = 0.0
    df.loc[df["wkswork2"] == 1, "r_wkswrkd"] = 7.0
    df.loc[df["wkswork2"] == 2, "r_wkswrkd"] = 20.0
    df.loc[df["wkswork2"] == 3, "r_wkswrkd"] = 33.0
    df.loc[df["wkswork2"] == 4, "r_wkswrkd"] = 43.5
    df.loc[df["wkswork2"] == 5, "r_wkswrkd"] = 48.5
    df.loc[df["wkswork2"] == 6, "r_wkswrkd"] = 51.0
    df["r_hrswrkd"] = df["r_wkswrkd"] * df["uhrswork"]
    df["r_loghrswrkd"] = np.where(df["r_hrswrkd"] > 0, np.log(df["r_hrswrkd"]), np.nan)
    df["r_lfp"] = (df["r_loghrswrkd"].notna()).astype(int)
    df["r_exp"] = df["age"] - df["r_yrsedu"] - 6
    df["r_exp2"] = df["r_exp"] ** 2

    # Identify same-sex couples using sploc
    df = df[df["sploc"] > 0].copy()
    df = add_spouse_characteristics(df)

    df["sscouple_all"] = (
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 201) & (df["sp_related"] == 101)) |
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 1114) & (df["sp_related"] == 101)) |
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 101) & (df["sp_related"] == 201)) |
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 101) & (df["sp_related"] == 1114))
    )

    df["sscouple_mar"] = (
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 201) & (df["sp_related"] == 101)) |
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 101) & (df["sp_related"] == 201))
    )

    df["sscouple_coh"] = (
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 1114) & (df["sp_related"] == 101)) |
        ((df["sex"] == df["sp_sex"]) & (df["r_related"] == 101) & (df["sp_related"] == 1114))
    )

    df["mflag"] = ((df["sscouple_all"]) & (df["r_related"] == 1114) & (df["qrelate"] == 9))
    pair_mflag = df["mflag"] | (((df["sscouple_all"]) & (df["sp_related"] == 1114) & (df["sp_qrelate"] == 9)))

    df.loc[pair_mflag & df["sscouple_all"], "sscouple_mar"] = True
    df.loc[pair_mflag & df["sscouple_all"], "sscouple_coh"] = False

    df["hh_with_ssc"] = df.groupby(["year", "serial"])["sscouple_all"].transform("max").astype(int)
    df["hh_with_ssmc"] = df.groupby(["year", "serial"])["sscouple_mar"].transform("max").astype(int)
    df["hh_with_sscc"] = df.groupby(["year", "serial"])["sscouple_coh"].transform("max").astype(int)

    # Identify same-sex cohabiters regardless of relationship status
    df["adult"] = ((df["age"] >= 18) | (df["sploc"] != 0)).astype(int)
    df["num_adults"] = df.groupby(["year", "serial"])["adult"].transform("sum")

    adults = df.loc[df["adult"] == 1, ["year", "serial", "pernum"]].copy()
    adults["adult_rank"] = adults.groupby(["year", "serial"]).cumcount() + 1
    adults2 = adults.groupby(["year", "serial"]).size().rename("n_adults").reset_index()
    adults = adults.merge(adults2, on=["year", "serial"], how="left")

    adult_pairs = adults[adults["n_adults"] == 2].groupby(["year", "serial"])["pernum"].agg(list).reset_index()
    pair_map = {}
    for _, row in adult_pairs.iterrows():
        a, b = row["pernum"]
        pair_map[(row["year"], row["serial"], a)] = b
        pair_map[(row["year"], row["serial"], b)] = a

    df["adloc"] = [
        pair_map.get((y, s, p), 0)
        for y, s, p in zip(df["year"], df["serial"], df["pernum"])
    ]

    # need other adult's sex for sscoh_coh
    ad = df[["year", "serial", "pernum", "sex", "adult"]].rename(
        columns={"pernum": "ad_pernum", "sex": "ad_sex", "adult": "ad_adult"}
    )
    df = df.merge(
        ad,
        left_on=["year", "serial", "adloc"],
        right_on=["year", "serial", "ad_pernum"],
        how="left"
    )

    df["sscoh_mar"] = df["sscouple_mar"].astype(int)
    df["sscoh_coh"] = (
        df["sscouple_coh"] |
        (
            (df["sex"] == df["ad_sex"]) &
            (df["adult"] == 1) &
            (df["ad_adult"] == 1) &
            (df["num_adults"] == 2) &
            (df["sploc"] == 0)
        )
    )
    df["sscoh_all"] = (df["sscoh_mar"] | df["sscoh_coh"]).astype(int)
    df["hh_with_sscoh"] = df.groupby(["year", "serial"])["sscoh_all"].transform("max").astype(int)
    df["hh_with_ssmcoh"] = df.groupby(["year", "serial"])["sscoh_mar"].transform("max").astype(int)
    df["hh_with_ssccoh"] = df.groupby(["year", "serial"])["sscoh_coh"].transform("max").astype(int)

    # Spouse characteristics for quasi-cohabiters using adloc if needed
    alt_cols = [
        "year", "serial", "pernum",
        "r_related", "r_relate", "r_race", "r_yrsedu", "r_edugroup",
        "r_wkswrkd", "r_hrswrkd", "r_loghrswrkd", "r_male", "r_female",
        "r_logincearn", "r_posincearn", "r_incearn", "r_incnonlabor", "r_inctot",
        "r_incwage", "r_incbus00", "r_incinvst", "r_incretir", "r_incss",
        "r_incwelfr", "r_incsupp", "r_incother", "r_lfp", "r_occbroad",
        "r_degbroad", "r_agegroup"
    ]
    alt = df[alt_cols].rename(columns={
        "pernum": "alt_pernum",
        "r_related": "alt_related",
        "r_relate": "alt_relate",
        "r_race": "alt_race",
        "r_yrsedu": "alt_yrsedu",
        "r_edugroup": "alt_edugroup",
        "r_wkswrkd": "alt_wkswrkd",
        "r_hrswrkd": "alt_hrswrkd",
        "r_loghrswrkd": "alt_loghrswrkd",
        "r_male": "alt_male",
        "r_female": "alt_female",
        "r_logincearn": "alt_logincearn",
        "r_posincearn": "alt_posincearn",
        "r_incearn": "alt_incearn",
        "r_incnonlabor": "alt_incnonlabor",
        "r_inctot": "alt_inctot",
        "r_incwage": "alt_incwage",
        "r_incbus00": "alt_incbus00",
        "r_incinvst": "alt_incinvst",
        "r_incretir": "alt_incretir",
        "r_incss": "alt_incss",
        "r_incwelfr": "alt_incwelfr",
        "r_incsupp": "alt_incsupp",
        "r_incother": "alt_incother",
        "r_lfp": "alt_lfp",
        "r_occbroad": "alt_occbroad",
        "r_degbroad": "alt_degbroad",
        "r_agegroup": "alt_agegroup",
    })

    df = df.merge(
        alt,
        left_on=["year", "serial", "adloc"],
        right_on=["year", "serial", "alt_pernum"],
        how="left"
    )

    no_ssc_but_ssccoh = (df["hh_with_ssc"] == 0) & (df["hh_with_ssccoh"] == 1)
    fill_cols = [
        ("sp_related", "alt_related"), ("sp_relate", "alt_relate"), ("sp_race", "alt_race"),
        ("sp_yrsedu", "alt_yrsedu"), ("sp_edugroup", "alt_edugroup"),
        ("sp_wkswrkd", "alt_wkswrkd"), ("sp_hrswrkd", "alt_hrswrkd"), ("sp_loghrswrkd", "alt_loghrswrkd"),
        ("sp_male", "alt_male"), ("sp_female", "alt_female"), ("sp_logincearn", "alt_logincearn"),
        ("sp_posincearn", "alt_posincearn"), ("sp_incearn", "alt_incearn"), ("sp_incnonlabor", "alt_incnonlabor"),
        ("sp_inctot", "alt_inctot"), ("sp_incwage", "alt_incwage"), ("sp_incbus00", "alt_incbus00"),
        ("sp_incinvst", "alt_incinvst"), ("sp_incretir", "alt_incretir"), ("sp_incss", "alt_incss"),
        ("sp_incwelfr", "alt_incwelfr"), ("sp_incsupp", "alt_incsupp"), ("sp_incother", "alt_incother"),
        ("sp_lfp", "alt_lfp"), ("sp_occbroad", "alt_occbroad"), ("sp_degbroad", "alt_degbroad"),
        ("sp_agegroup", "alt_agegroup")
    ]
    for dst, src in fill_cols:
        df.loc[no_ssc_but_ssccoh, dst] = df.loc[no_ssc_but_ssccoh, src]

    # Earner status
    df["r_earnstatus"] = np.nan
    df.loc[(df["r_incearn"] >= df["sp_incearn"]) & df["sp_incearn"].notna(), "r_earnstatus"] = 1
    df.loc[(df["r_incearn"] < df["sp_incearn"]) & df["sp_incearn"].notna(), "r_earnstatus"] = 2

    tie_ssc = (
        (df["r_incearn"] == df["sp_incearn"]) &
        df["sp_incearn"].notna() &
        (df["r_related"] != 101) &
        (df["sscouple_all"])
    )
    tie_sscoh = (
        (df["r_incearn"] == df["sp_incearn"]) &
        df["sp_incearn"].notna() &
        (df["pernum"] < df["adloc"]) &
        (~df["sscouple_all"]) &
        (df["sscoh_all"] == 1)
    )
    df.loc[tie_ssc | tie_sscoh, "r_earnstatus"] = 2

    non_partner = (
        (
            ~df["r_related"].isin([101, 201, 1114]) &
            (df["hh_with_ssc"] == 1)
        ) |
        (
            (df["adult"] == 0) &
            (df["sscouple_mar"] == 0) &
            (df["sscouple_coh"] == 0)
        )
    )
    df.loc[non_partner, "r_earnstatus"] = 0

    # Dependents / children
    df["dependent"] = (
        ((df["age"] <= 18) & (df["momloc"] != 0)) |
        ((df["age"] <= 18) & (df["poploc"] != 0)) |
        ((df["age"] <= 23) & (df["momloc"] != 0) & (df["school"] == 2)) |
        ((df["age"] <= 23) & (df["poploc"] != 0) & (df["school"] == 2)) |
        ((df["age"] < 17) & (df["momloc"] == 0) & (df["poploc"] == 0) & (df["momloc2"] == 0) & (df["poploc2"] == 0))
    ).astype(int)

    sscouple_lookup = df[["year", "serial", "pernum", "sscouple_all", "sscoh_all"]].copy()

    mom_lookup = sscouple_lookup.rename(columns={
        "pernum": "momloc",
        "sscouple_all": "mom_sscouple_all",
        "sscoh_all": "mom_sscoh_all"
    })
    pop_lookup = sscouple_lookup.rename(columns={
        "pernum": "poploc",
        "sscouple_all": "pop_sscouple_all",
        "sscoh_all": "pop_sscoh_all"
    })

    df = df.merge(mom_lookup, on=["year", "serial", "momloc"], how="left")
    df = df.merge(pop_lookup, on=["year", "serial", "poploc"], how="left")

    bad_dep = (
        ((df["momloc"] != 0) & (df["mom_sscouple_all"].fillna(0) == 0)) |
        ((df["poploc"] != 0) & (df["pop_sscouple_all"].fillna(0) == 0)) |
        ((df["momloc"] != 0) & (df["mom_sscoh_all"].fillna(0) == 0)) |
        ((df["poploc"] != 0) & (df["pop_sscoh_all"].fillna(0) == 0))
    )
    df.loc[bad_dep, "dependent"] = 0

    df["c_children"] = df.groupby(["year", "serial"])["dependent"].transform("sum")
    df["c_anychildren"] = (df["c_children"] > 0).astype(int)
    df["r_child0_1"] = ((df["dependent"] == 1) & (df["age"] == 0)).astype(int)
    df["r_child1_5"] = ((df["dependent"] == 1) & (df["age"].between(1, 5))).astype(int)
    df["r_child6_18"] = ((df["dependent"] == 1) & (df["age"].between(6, 18))).astype(int)
    df["r_child0_12"] = ((df["dependent"] == 1) & (df["age"].between(0, 12))).astype(int)
    df["r_child0_16"] = ((df["dependent"] == 1) & (df["age"].between(0, 16))).astype(int)
    df["r_child0_18"] = ((df["dependent"] == 1) & (df["age"].between(0, 18))).astype(int)

    for src, dst in [
        ("r_child0_1", "c_children0_1"),
        ("r_child1_5", "c_children1_5"),
        ("r_child6_18", "c_children6_18"),
        ("r_child0_12", "c_children0_12"),
        ("r_child0_16", "c_children0_16"),
        ("r_child0_18", "c_children0_18"),
    ]:
        df[dst] = df.groupby(["year", "serial"])[src].transform("sum")

    # Couple vars
    couple_mask = (
        df["r_related"].isin([101, 201, 1114]) |
        ((df["adult"] == 1) & (df["sscoh_coh"] == 1))
    )

    df["c_agemax"] = df["age"].where(couple_mask).groupby([df["year"], df["serial"]]).transform("max")
    df["c_agemin"] = df["age"].where(couple_mask).groupby([df["year"], df["serial"]]).transform("min")
    df["c_agediff"] = df["c_agemax"] - df["c_agemin"]

    df["c_edumax"] = df["r_yrsedu"].where(couple_mask).groupby([df["year"], df["serial"]]).transform("max")
    df["c_edumin"] = df["r_yrsedu"].where(couple_mask).groupby([df["year"], df["serial"]]).transform("min")
    df["c_edudiff"] = df["c_edumax"] - df["c_edumin"]

    df["c_dualearner"] = (
        ((df["r_posincearn"] == 1) & (df["sp_posincearn"] == 1)) & couple_mask
    ).astype(int)
    df["c_singleearner"] = (
        (((df["r_posincearn"] == 1) & (df["sp_posincearn"] == 0)) |
         ((df["r_posincearn"] == 0) & (df["sp_posincearn"] == 1))) & couple_mask
    ).astype(int)
    df["c_noearner"] = (
        ((df["r_posincearn"] == 0) & (df["sp_posincearn"] == 0)) & couple_mask
    ).astype(int)

    df["c_racecomp"] = (df["r_race"] == df["sp_race"]).astype(int)
    df["c_incearn"] = (
        df["r_incwage"].fillna(0) +
        df["r_incbus00"].fillna(0) +
        df["sp_incwage"].fillna(0) +
        df["sp_incbus00"].fillna(0)
    )

    for i in range(1, 6):
        df[f"c_incearn{i}"] = (df["c_incearn"] / 10000) ** i

    df["c_incearn_split"] = df["r_incearn"] / df["c_incearn"]
    df.loc[df["r_earnstatus"] == 2, "c_incearn_split"] = (
        df.loc[df["r_earnstatus"] == 2, "sp_incearn"] /
        df.loc[df["r_earnstatus"] == 2, "c_incearn"]
    )
    df["c_incearn_split"] = df["c_incearn_split"].replace([np.inf, -np.inf], np.nan).fillna(0)

    for i in range(1, 6):
        df[f"c_incearn_split{i}"] = df["c_incearn_split"] ** i

    df["c_incearn_diff"] = np.where(
        df["r_earnstatus"] == 1,
        df["r_incwage"] + df["r_incbus00"] - df["sp_incwage"] - df["sp_incbus00"],
        df["sp_incwage"] + df["sp_incbus00"] - df["r_incwage"] - df["r_incbus00"]
    )

    df["c_relate"] = np.nan
    df.loc[df["sscouple_mar"] == 1, "c_relate"] = 1
    df.loc[df["sscouple_coh"] == 1, "c_relate"] = 2
    df.loc[((df["r_related"] == 101) & df["sp_related"].between(301, 601)) |
           ((df["sp_related"] == 101) & df["r_related"].between(301, 601)), "c_relate"] = 3
    df.loc[((df["r_related"] == 101) & df["sp_related"].between(701, 1001)) |
           ((df["sp_related"] == 101) & df["r_related"].between(701, 1001)), "c_relate"] = 4
    df.loc[((df["r_related"] == 101) & df["sp_related"].between(1115, 1241)) |
           ((df["sp_related"] == 101) & df["r_related"].between(1115, 1241)), "c_relate"] = 5
    df.loc[((df["r_related"] == 101) & df["sp_related"].between(1242, 1260)) |
           ((df["sp_related"] == 101) & df["r_related"].between(1242, 1260)), "c_relate"] = 6

    df = add_policy_variables(df)

    # Keep one row per couple: household head
    df_final = df[(df["sscouple_all"]) & (df["r_related"] == 101)].copy()

    # Apply paper age restriction
    df_final = df_final[(df_final["c_agemin"] >= 18) & (df_final["c_agemax"] <= 60)].copy()

    keep_cols = [
        "year", "serial", "statefip", "countyfip", "hhwt",
        "sscouple_mar", "sscouple_coh", "sscouple_all",
        "sex", "age", "sp_age", "r_male",
        "r_incearn", "sp_incearn",
        "r_incwage", "r_incbus00", "sp_incwage", "sp_incbus00",
        "r_incinvst", "r_incretir", "r_incss", "r_incwelfr", "r_incsupp", "r_incother",
        "sp_incinvst", "sp_incretir", "sp_incss", "sp_incwelfr", "sp_incsupp", "sp_incother",
        "r_yrsedu", "sp_yrsedu", "c_edumax", "c_edumin", "c_edudiff",
        "r_race", "sp_race", "c_racecomp",
        "r_agegroup", "sp_agegroup",
        "r_edugroup", "sp_edugroup",
        "r_occbroad", "sp_occbroad",
        "r_degbroad", "sp_degbroad",
        "r_earnstatus", "r_posincearn", "sp_posincearn",
        "c_agemax", "c_agemin", "c_agediff",
        "c_children", "c_anychildren",
        "c_children0_1", "c_children1_5", "c_children6_18",
        "c_children0_12", "c_children0_16", "c_children0_18",
        "c_dualearner", "c_singleearner", "c_noearner",
        "c_incearn", "c_incearn1", "c_incearn2", "c_incearn3", "c_incearn4", "c_incearn5",
        "c_incearn_split", "c_incearn_split1", "c_incearn_split2", "c_incearn_split3", "c_incearn_split4", "c_incearn_split5",
        "c_incearn_diff", "c_relate",
        "mortgage", "mortgag2", "mortamt1", "mortamt2", "taxincl", "insincl", "propinsr", "proptx99", "rent",
        "migrate1", "migrate1d", "migplac1", "hinsemp",
        "staterecog", "staterecog2", "staterecog_policy", "staterecog_time", "medicaid_exp",
        "preW", "postWpreO", "postO"
    ]

    keep_cols = [c for c in keep_cols if c in df_final.columns]
    df_final = df_final[keep_cols].copy()

    print("Final output shape:", df_final.shape)
    print("Share married:", df_final["sscouple_mar"].mean())

    df_final.to_pickle(OUTPUT_FILE)
    print(f"Saved final file to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()