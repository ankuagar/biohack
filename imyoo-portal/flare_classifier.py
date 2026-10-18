"""Can a model trained on ImYoo's samples tell flare blood from calm blood in people it has never seen?

Each blood sample is a bag of thousands of cells with one label (flare or calm). We summarise each sample
(cell-type mix, or gene activity per cell type), train simple models, and test every one with
leave-one-person-out: the person being scored is never in the training data. A shuffle test (flare labels
scrambled within each person, model retrained) shows whether a result beats chance.

Run INSIDE SageMaker after precompute.py (no big file needed; ~10-25 minutes):
    python flare_classifier.py
Writes data/classifier/results.json and predictions.parquet for the portal.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from precompute import family_of

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JUNK = "artifact|Debris|Multiplet|Unknown|Unlabelled|Platelet"
TOP_GENES = 200       # gene features chosen inside each training round (never using the held-out person)
MAX_CELL_TYPES = 8    # most common cell types used for gene-activity features
C = 0.1               # strength of the model's restraint (smaller = simpler model)




def mix_features(data, samples):
    """Cell-type mix per sample: families plus main cell types, as log shares."""
    c = pd.read_parquet(f"{data}/composition.parquet")
    c = c[~c["cell_type"].str.contains(JUNK, case=False)]
    b = c[c["level"] == "broad"].copy()
    fam = b.assign(cell_type=b["cell_type"].map(family_of)).groupby(["sample_id", "cell_type"])["pct"].sum()
    fam = fam.unstack().add_prefix("family: ")
    main = b.pivot_table(index="sample_id", columns="cell_type", values="pct", fill_value=0).add_prefix("type: ")
    X = fam.join(main, how="outer").reindex(samples.index).fillna(0)
    X = X.loc[:, ~X.columns.duplicated()]
    return np.log(X + 0.1)


def gene_features(data, samples):
    """Gene activity (log counts per million) per sample, for the most common cell types."""
    pb = pd.read_parquet(f"{data}/pseudobulk.parquet")
    pb = pb[pb["sample_id"].isin(samples.index)]
    coverage = pb.groupby("cell_type")["sample_id"].nunique() / len(samples)
    common = pb.groupby("cell_type")["n_cells"].mean()[coverage >= 0.8].sort_values(ascending=False)
    types = list(common.index[:MAX_CELL_TYPES])
    genes = [g for g in pb.columns if g not in ("sample_id", "cell_type", "n_cells")]
    parts = []
    for t in types:
        part = pb[pb["cell_type"] == t].set_index("sample_id")[genes]
        parts.append(part.add_prefix(f"{t}: "))
    X = pd.concat(parts, axis=1).reindex(samples.index)
    return X.astype("float32"), types


def personal(X, samples):
    """Each sample minus the same person's OTHER calm samples (never itself). Rows without a reference -> NaN."""
    V = X.to_numpy(dtype="float64")
    out = np.full(V.shape, np.nan)
    persons = samples.loc[X.index, "person"].to_numpy()
    calm = (samples.loc[X.index, "flare"] == "Baseline").to_numpy()
    for p in np.unique(persons):
        idx = np.where(persons == p)[0]
        for i in idx:
            ref = [j for j in idx if j != i and calm[j]]
            if ref:
                out[i] = V[i] - np.nanmean(V[ref], axis=0)
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def leave_one_person_out(X, y, persons, top=None, rng=None):
    """Score every sample with a model trained without that sample's person."""
    X = X.to_numpy(dtype="float64")
    score = np.full(len(y), np.nan)
    for p in np.unique(persons):
        tr, te = persons != p, persons == p
        if len(set(y[tr])) < 2:
            continue
        Xtr, Xte = X[tr], X[te]
        mu = np.nanmean(Xtr, axis=0)
        mu = np.where(np.isnan(mu), 0, mu)
        Xtr, Xte = np.where(np.isnan(Xtr), mu, Xtr), np.where(np.isnan(Xte), mu, Xte)
        if top and Xtr.shape[1] > top:  # pick features using the training people only
            a, b = Xtr[y[tr] == 1], Xtr[y[tr] == 0]
            t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-9)
            keep = np.argsort(-np.abs(t))[:top]
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        sd = Xtr.std(0) + 1e-6
        m = Xtr.mean(0)
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        model.fit((Xtr - m) / sd, y[tr])
        score[te] = model.predict_proba((Xte - m) / sd)[:, 1]
    return score


def summarise(score, y, persons):
    ok = ~np.isnan(score)
    auc = roc_auc_score(y[ok], score[ok]) if len(set(y[ok])) == 2 else float("nan")
    higher = total = 0
    for p in np.unique(persons[ok]):
        m = ok & (persons == p)
        if len(set(y[m])) == 2:
            total += 1
            higher += int(score[m & (y == 1)].mean() > score[m & (y == 0)].mean())
    return auc, higher, total, int(ok.sum())


def shuffle_within_person(y, persons, rng):
    y2 = y.copy()
    for p in np.unique(persons):
        m = np.where(persons == p)[0]
        y2[m] = rng.permutation(y2[m])
    return y2


def top_features(X, y, top=None, n=15):
    """Which features the model leans on, fitted once on everyone (for explanation only)."""
    Xv = X.to_numpy(dtype="float64")
    ok = ~np.isnan(Xv).all(axis=1)
    Xv, yv = Xv[ok], y[ok]
    mu = np.nanmean(Xv, axis=0)
    mu = np.where(np.isnan(mu), 0, mu)
    Xv = np.where(np.isnan(Xv), mu, Xv)
    cols = np.array(X.columns)
    if top and Xv.shape[1] > top:
        a, b = Xv[yv == 1], Xv[yv == 0]
        t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-9)
        keep = np.argsort(-np.abs(t))[:top]
        Xv, cols = Xv[:, keep], cols[keep]
    sd = Xv.std(0) + 1e-6
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit((Xv - Xv.mean(0)) / sd, yv)
    w = pd.Series(model.coef_[0], index=cols)
    w = w.reindex(w.abs().sort_values(ascending=False).index[:n])
    return [{"feature": k, "weight": float(v), "points_to": "flare" if v > 0 else "calm"} for k, v in w.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--shuffles", type=int, default=100, help="shuffle-test repeats per approach")
    ap.add_argument("--group", default="Bowel disease")
    args = ap.parse_args()
    out = f"{args.data}/classifier"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)

    def say(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    samples = pd.read_parquet(f"{args.data}/samples.parquet").set_index("sample_id")
    if args.group != "Everyone":
        samples = samples[samples["group"] == args.group]
    y = (samples["flare"] == "Flaring").astype(int).to_numpy()
    persons = samples["person"].to_numpy()
    say(f"{len(samples)} samples from {samples['person'].nunique()} people ({args.group}); "
        f"{y.sum()} flare, {len(y) - y.sum()} calm")

    mix = mix_features(args.data, samples)
    genes, types = gene_features(args.data, samples)
    say(f"features: cell mix {mix.shape[1]}, gene activity {genes.shape[1]:,} ({len(types)} cell types: {', '.join(types)})")
    approaches = {
        "Cell mix": (mix, None),
        "Gene activity": (genes, TOP_GENES),
        "Cell mix vs own normal": (personal(mix, samples), None),
        "Gene activity vs own normal": (personal(genes, samples), TOP_GENES),
    }
    results, preds = {}, []
    for name, (X, top) in approaches.items():
        score = leave_one_person_out(X, y, persons, top)
        auc, higher, total, n = summarise(score, y, persons)
        null = []
        for _ in range(args.shuffles):
            ys = shuffle_within_person(y, persons, rng)
            null.append(summarise(leave_one_person_out(X, ys, persons, top), ys, persons)[0])
        null = np.array([v for v in null if v == v])
        p = (1 + (null >= auc).sum()) / (1 + len(null)) if len(null) else float("nan")
        results[name] = {"correct_ranking": auc, "people_flare_higher": higher, "people_with_both": total,
                         "samples_scored": n, "shuffle_p": p, "shuffle_mean": float(null.mean()) if len(null) else None,
                         "shuffle_95th": float(np.percentile(null, 95)) if len(null) else None,
                         "features": int(X.shape[1]), "features_used_per_round": top or int(X.shape[1]),
                         "top_features": top_features(X, y, top)}
        preds.append(pd.DataFrame({"approach": name, "sample_id": samples.index, "person": persons,
                                   "flare": samples["flare"].to_numpy(), "score": score}))
        say(f"{name:28s} correct ranking {auc:.0%} | flare scored higher in {higher}/{total} people | "
            f"shuffle-test p = {p:.3f} (shuffled average {results[name]['shuffle_mean']:.0%})")
    info = {"built": time.strftime("%Y-%m-%d %H:%M"), "group": args.group, "samples": int(len(samples)),
            "people": int(samples["person"].nunique()), "shuffles": args.shuffles, "cell_types_for_genes": types,
            "top_genes_per_round": TOP_GENES, "results": results}
    with open(f"{out}/results.json", "w") as f:
        json.dump(info, f, indent=2)
    pd.concat(preds, ignore_index=True).to_parquet(f"{out}/predictions.parquet", index=False)
    say(f"Done -> {out}/results.json")


if __name__ == "__main__":
    main()
