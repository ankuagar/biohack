"""Make MADE-UP data in the same shape as precompute.py's output, to try the portal anywhere.

    python make_demo_data.py && streamlit run app.py
No real ImYoo data is used.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from precompute import OUT, build_samples, composition, gene_tests

rng = np.random.default_rng(42)

# detailed type -> (broad family, T-cell subset options, typical share)
TYPES = {
    "CD4 T Cells": ("T Cells", ["CD4 Naive T Cells", "CD4 Memory T Cells", "CD4 Regulatory T Cells"], 30),
    "CD8 T Cells": ("T Cells", ["CD8 Naive T Cells", "CD8 Memory T Cells", "CD8 Effector T Cells"], 18),
    "MAIT Cells": ("T Cells", ["MAIT Cells"], 3),
    "Gamma Delta T Cells": ("T Cells", ["Gamma Delta T Cells"], 3),
    "CD56 Dim NK Cells": ("NK Cells", None, 12),
    "CD56 Bright NK Cells": ("NK Cells", None, 2),
    "Naive B Cells": ("B Cells", None, 7),
    "Memory B Cells": ("B Cells", None, 4),
    "Plasmablasts": ("B Cells", None, 0.6),
    "Classical Monocytes": ("Monocytes", None, 4),
    "Non-classical Monocytes": ("Monocytes", None, 1.2),
    "Conventional Dendritic Cells": ("Dendritic Cells", None, 0.8),
    "Plasmacytoid Dendritic Cells": ("Dendritic Cells", None, 0.4),
    "Proliferating T Cells": ("Proliferating Lymphoid Cells", None, 0.3),
}
FLARE_EFFECT = {"Classical Monocytes": 1.6, "Plasmablasts": 2.0, "Naive B Cells": 0.75, "CD4 Regulatory T Cells": 0.8}
DISEASES = ["Ulcerative Colitis"] * 9 + ["Crohn's Disease"] * 8 + ["Crohn's Colitis|Crohn's Disease"] * 2 + [
    "Crohn's Disease|Indeterminate Colitis", "Fibromyalgia|Crohn's Disease", "Ibs|Ibd|Celiac Disease",
    "Eczema|Crohn's Disease|Anxiety|Gerd", "Collagenous Colitis", "Rheumatoid Arthritis", "Fibromyalgia",
    "Peripheral Spondyloarthritis", "Rheumatoid Arthritis|Irritable Bowel Syndrome",
    "Lupus-acquired Von Willebrand Disease (vwd)"]
GENES = ["S100A8", "S100A9", "S100A12", "LYZ", "CD14", "FCGR3A", "IL1B", "TNF", "CXCL8", "IFITM3", "ISG15", "MX1",
         "IFI44L", "JCHAIN", "MZB1", "IGHG1", "IGHA1", "XBP1", "CD79A", "MS4A1", "CD19", "TCL1A", "IL7R", "CCR7",
         "SELL", "LEF1", "TCF7", "FOXP3", "IL2RA", "CTLA4", "GZMB", "GZMK", "PRF1", "NKG7", "GNLY", "KLRB1",
         "CD3E", "CD4", "CD8A", "MKI67", "TOP2A", "HLA-DRA", "CD74", "FCER1A", "CLEC10A", "LILRA4", "IRF7",
         "ACTB", "B2M", "MALAT1"] + [f"GENE{i:03d}" for i in range(250)]
UP_GENES = {"S100A8", "S100A9", "S100A12", "IL1B", "ISG15", "MX1", "IFI44L", "JCHAIN", "MZB1", "MKI67"}
DOWN_GENES = {"CCR7", "SELL", "TCL1A", "FOXP3"}


def fake_obs():
    rows = []
    run_names = ["R93", "R96", "R97", "R99", "R100"]
    for i, disease in enumerate(DISEASES):
        pid = f"{rng.integers(0, 16**6):06x}"
        n = int(rng.integers(2, 8))
        states = rng.permutation(["Flaring"] * max(1, n // 2) + ["Baseline"] * (n - max(1, n // 2)))
        if i % 7 == 6:  # a few people with only one kind of sample
            states = ["Baseline"] * n
        days = np.sort(rng.uniform(0, 240, n))
        run = rng.choice(run_names)
        age = float(rng.integers(19, 70))
        person_bias = {t: rng.lognormal(0, 0.25) for t in TYPES}
        for j, (state, day) in enumerate(zip(states, days)):
            weights = np.array([TYPES[t][2] * person_bias[t] * (FLARE_EFFECT.get(t, 1) if state == "Flaring" else 1)
                                for t in TYPES])
            n_cells = int(rng.integers(250, 400))
            steroid = float(rng.uniform(2, 200)) if state == "Flaring" and rng.random() < 0.4 else np.nan
            picks = rng.choice(list(TYPES), size=n_cells, p=weights / weights.sum())
            for t in picks:
                broad, subs, _ = TYPES[t]
                rows.append((pid, f"{pid}-CC-{j + 1}", state, disease, run, day * 86400,
                             float(rng.uniform(4, 200)), float(rng.integers(0, 15)), broad, t,
                             rng.choice(subs) if subs else None, age, steroid))
    obs = pd.DataFrame(rows, columns=[
        "participant_id", "sample_id", "flare_status", "canonical_disease", "cell_barcoding_run_id",
        "relative_timepoint_seconds", "cell_barcoding_delay_days", "harvey_bradshaw_index",
        "cell_type_level_2", "cell_type_level_3", "cell_type_level_4", "age_at_start_years", "hours_since_steroids"])
    # one freezer time / score per sample, like the real data
    for c in ["cell_barcoding_delay_days", "harvey_bradshaw_index", "hours_since_steroids"]:
        obs[c] = obs.groupby("sample_id")[c].transform("first")
    return obs


def fake_map(obs, samples):
    centers = {t: rng.normal(0, 8, 2) for t in TYPES}
    for t, (broad, _, _) in TYPES.items():  # keep families near each other
        fam = [k for k, v in TYPES.items() if v[0] == broad][0]
        centers[t] = centers[fam] + rng.normal(0, 2.0, 2) if t != fam else centers[t]
    sub = obs.sample(min(20_000, len(obs)), random_state=0)
    xy = np.vstack([centers[t] + rng.normal(0, 0.9, 2) for t in sub["cell_type_level_3"]])
    person = dict(zip(samples["sample_id"], samples["person"]))
    group = dict(zip(samples["sample_id"], samples["group"]))
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "broad": sub["cell_type_level_2"].values,
                         "detailed": sub["cell_type_level_3"].values, "flare": sub["flare_status"].values,
                         "person": sub["sample_id"].map(person).values, "group": sub["sample_id"].map(group).values})


def fake_pseudobulk(samples):
    base = {g: rng.uniform(2, 11) for g in GENES}
    person_shift = {p: rng.normal(0, 0.3) for p in samples["person"].unique()}
    rows = []
    for _, s in samples.iterrows():
        for ct in sorted({v[0] for v in TYPES.values()}):
            vals = {}
            for g in GENES:
                v = base[g] + person_shift[s["person"]] + rng.normal(0, 0.35)
                if s["flare"] == "Flaring":
                    v += 0.9 if g in UP_GENES else -0.7 if g in DOWN_GENES else 0
                vals[g] = v
            rows.append({"sample_id": s["sample_id"], "cell_type": ct, "n_cells": int(rng.integers(20, 300)), **vals})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    obs = fake_obs()
    samples = build_samples(obs)
    samples.to_parquet(f"{OUT}/samples.parquet", index=False)
    composition(obs, samples).to_parquet(f"{OUT}/composition.parquet", index=False)
    cells = fake_map(obs, samples)
    cells.to_parquet(f"{OUT}/cells_map.parquet", index=False)
    pb = fake_pseudobulk(samples)
    pb.to_parquet(f"{OUT}/pseudobulk.parquet", index=False)
    long = pb.drop(columns="n_cells").melt(id_vars=["sample_id", "cell_type"], var_name="gene", value_name="logcpm")
    gene_tests(long, samples).to_parquet(f"{OUT}/gene_tests.parquet", index=False)
    removed = int(len(obs) * 0.11)
    with open(f"{OUT}/info.json", "w") as f:
        json.dump({"demo": True, "built": time.strftime("%Y-%m-%d %H:%M"), "cells_total": len(obs) + removed,
                   "cells_removed": removed, "cells_kept": len(obs), "genes_total": 36601,
                   "people": int(samples["person"].nunique()), "samples": len(samples),
                   "map_source": "made-up demo layout", "map_cells": len(cells)}, f, indent=2)
    print(f"Demo data written to {OUT}: {len(samples)} samples, {samples['person'].nunique()} people, {len(obs):,} cells")


if __name__ == "__main__":
    main()
