import os
import gc
import glob
import pandas as pd

RAW_FILE = "Data/Raw/acs_2012-2017.dat"

OUT_DIR = "Data/Intermediate/ssc_id_chunks"
FINAL_IDS = "Data/Intermediate/ssc_households_V2.pkl"
OUT_FILE = "Data/Intermediate/acs_ssc_reduced_V2.pkl"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# PASS 1: identify household IDs containing same-sex couples
# =========================================================

# Minimal schema needed to identify same-sex couples and mflag logic
id_colspecs = [
    (0, 4),      # year
    (6, 14),     # serial
    (77, 81),    # pernum
    (95, 96),    # subfam
    (97, 98),    # sfrelate
    (106, 108),  # sploc
    (124, 126),  # relate
    (126, 130),  # related
    (130, 131),  # sex
    (131, 134),  # age
    (269, 270),  # qrelate
]

id_names = [
    "year", "serial", "pernum", "subfam", "sfrelate",
    "sploc", "relate", "related", "sex", "age", "qrelate"
]

reader = pd.read_fwf(
    RAW_FILE,
    colspecs=id_colspecs,
    names=id_names,
    chunksize=200_000
)

saved_files = []

for i, chunk in enumerate(reader, start=1):
    chunk = chunk[chunk["year"].between(2012, 2017)].copy()

    for col in id_names:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Keep only households where someone has spouse/partner linkage
    candidate = chunk.groupby(["year", "serial"])["sploc"].transform(lambda s: (s > 0).any())
    chunk = chunk[candidate].copy()

    if chunk.empty:
        print(f"Chunk {i}: no candidate households")
        continue

    sp = chunk[["year", "serial", "pernum", "sex", "related", "qrelate"]].copy()
    sp = sp.rename(columns={
        "pernum": "sp_pernum",
        "sex": "sp_sex",
        "related": "sp_related",
        "qrelate": "sp_qrelate"
    })

    merged = chunk.merge(
        sp,
        left_on=["year", "serial", "sploc"],
        right_on=["year", "serial", "sp_pernum"],
        how="left",
        validate="many_to_one"
    )

    merged["sscouple_mar"] = (
        (merged["sex"] == merged["sp_sex"]) &
        (
            ((merged["related"] == 101) & (merged["sp_related"] == 201)) |
            ((merged["related"] == 201) & (merged["sp_related"] == 101))
        )
    )

    merged["sscouple_coh"] = (
        (merged["sex"] == merged["sp_sex"]) &
        (
            ((merged["related"] == 101) & (merged["sp_related"] == 1114)) |
            ((merged["related"] == 1114) & (merged["sp_related"] == 101))
        )
    )

    merged["sscouple_all"] = merged["sscouple_mar"] | merged["sscouple_coh"]

    # Replication package mflag logic:
    # mflag = (sscouple_all == 1 & related == 1114 & qrelate == 9)
    merged["mflag"] = (
        merged["sscouple_all"] &
        (merged["related"] == 1114) &
        (merged["qrelate"] == 9)
    )

    # Pair-level adjustment: if either partner has mflag, keep household
    pair_mflag = merged["mflag"] | (
        merged["sscouple_all"] &
        (merged["sp_related"] == 1114) &
        (merged["sp_qrelate"] == 9)
    )

    merged["ssc_keep"] = merged["sscouple_all"] | pair_mflag

    hh_keep = merged.groupby(["year", "serial"])["ssc_keep"].transform("max")
    keep = merged.loc[hh_keep == 1, ["year", "serial"]].drop_duplicates()

    out_path = os.path.join(OUT_DIR, f"ssc_ids_{i:03d}.pkl")
    keep.to_pickle(out_path)
    saved_files.append(out_path)

    print(f"Chunk {i}: kept {len(keep):,} households")

    del chunk, sp, merged, keep
    gc.collect()

files = sorted(glob.glob(os.path.join(OUT_DIR, "ssc_ids_*.pkl")))
ssc_ids = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True).drop_duplicates()

print("Unique same-sex-couple households:", len(ssc_ids))
ssc_ids.to_pickle(FINAL_IDS)
print("Saved household IDs to", FINAL_IDS)

# =========================================================
# PASS 2: read richer schema only for selected households
# =========================================================

ssc_ids = pd.read_pickle(FINAL_IDS)
ssc_set = set(map(tuple, ssc_ids[["year", "serial"]].to_numpy()))

# Rich schema aligned to read.do + what clean.do / tax stage need
colspecs = [
    (0, 4),      # year
    (6, 14),     # serial
    (27, 37),    # hhwt
    (37, 39),    # statefip
    (39, 42),    # countyfip
    (49, 50),    # mortgage
    (50, 51),    # mortgag2
    (51, 56),    # mortamt1
    (56, 60),    # mortamt2
    (60, 61),    # taxincl
    (61, 62),    # insincl
    (62, 66),    # propinsr
    (66, 68),    # proptx99
    (68, 72),    # rent
    (77, 81),    # pernum
    (91, 93),    # famunit
    (93, 95),    # famsize
    (95, 96),    # subfam
    (96, 97),    # sftype
    (97, 98),    # sfrelate
    (98, 100),   # momloc
    (100, 102),  # momrule
    (102, 104),  # poploc
    (104, 106),  # poprule
    (106, 108),  # sploc
    (108, 110),  # sprule
    (110, 112),  # momloc2
    (112, 114),  # mom2rule
    (114, 116),  # poploc2
    (116, 118),  # pop2rule
    (118, 119),  # nchild
    (119, 120),  # nchlt5
    (120, 122),  # eldch
    (122, 124),  # yngch
    (124, 126),  # relate
    (126, 130),  # related
    (130, 131),  # sex
    (131, 134),  # age
    (134, 135),  # marst
    (135, 139),  # birthyr
    (139, 140),  # marrno
    (140, 141),  # marrinyr
    (141, 145),  # yrmarr
    (145, 146),  # divinyr
    (146, 147),  # widinyr
    (147, 148),  # race
    (148, 151),  # raced
    (151, 152),  # hispan
    (152, 155),  # hispand
    (155, 156),  # hcovany
    (156, 157),  # hinsemp
    (157, 158),  # school
    (158, 160),  # educ
    (160, 163),  # educd
    (163, 165),  # degfield
    (165, 169),  # degfieldd
    (169, 171),  # degfield2
    (171, 175),  # degfield2d
    (175, 176),  # empstat
    (176, 178),  # empstatd
    (178, 179),  # labforce
    (179, 183),  # occ
    (183, 187),  # ind
    (187, 188),  # classwkr
    (188, 190),  # classwkrd
    (190, 191),  # wkswork2
    (191, 193),  # uhrswork
    (193, 200),  # inctot
    (200, 207),  # ftotinc
    (207, 213),  # incwage
    (213, 219),  # incbus00
    (219, 224),  # incss
    (224, 229),  # incwelfr
    (229, 235),  # incinvst
    (235, 241),  # incretir
    (241, 246),  # incsupp
    (246, 251),  # incother
    (251, 258),  # incearn
    (258, 261),  # poverty
    (261, 262),  # migrate1
    (262, 264),  # migrate1d
    (264, 267),  # migplac1
    (267, 268),  # movedin
    (268, 269),  # qmarst
    (269, 270),  # qrelate
    (270, 271),  # qsex
    (271, 272),  # qyrmarr
]

names = [
    "year", "serial", "hhwt", "statefip", "countyfip",
    "mortgage", "mortgag2", "mortamt1", "mortamt2", "taxincl", "insincl",
    "propinsr", "proptx99", "rent",
    "pernum", "famunit", "famsize", "subfam", "sftype", "sfrelate",
    "momloc", "momrule", "poploc", "poprule", "sploc", "sprule",
    "momloc2", "mom2rule", "poploc2", "pop2rule",
    "nchild", "nchlt5", "eldch", "yngch",
    "relate", "related", "sex", "age", "marst", "birthyr",
    "marrno", "marrinyr", "yrmarr", "divinyr", "widinyr",
    "race", "raced", "hispan", "hispand", "hcovany", "hinsemp",
    "school", "educ", "educd", "degfield", "degfieldd", "degfield2", "degfield2d",
    "empstat", "empstatd", "labforce", "occ", "ind", "classwkr", "classwkrd",
    "wkswork2", "uhrswork", "inctot", "ftotinc", "incwage", "incbus00",
    "incss", "incwelfr", "incinvst", "incretir", "incsupp", "incother",
    "incearn", "poverty", "migrate1", "migrate1d", "migplac1", "movedin",
    "qmarst", "qrelate", "qsex", "qyrmarr"
]

reader = pd.read_fwf(
    RAW_FILE,
    colspecs=colspecs,
    names=names,
    chunksize=200_000
)

parts = []
total = 0

for i, chunk in enumerate(reader, start=1):
    chunk = chunk[chunk["year"].between(2012, 2017)].copy()

    for col in names:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    keys = list(zip(chunk["year"], chunk["serial"]))
    mask = [k in ssc_set for k in keys]
    chunk = chunk.loc[mask].copy()

    if chunk.empty:
        continue

    chunk["hhwt"] = chunk["hhwt"] / 100
    parts.append(chunk)
    total += len(chunk)

    print(f"Chunk {i}: kept {len(chunk):,} rows | total {total:,}")
    del chunk
    gc.collect()

ssc = pd.concat(parts, ignore_index=True)
print("Reduced dataset shape:", ssc.shape)

ssc.to_pickle(OUT_FILE)
print("Saved reduced same-sex-couple microdata to", OUT_FILE)