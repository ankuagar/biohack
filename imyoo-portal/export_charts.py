"""Draw the portal's key charts as interactive HTML pieces, for a results page shared with named reviewers.

Run INSIDE SageMaker after precompute.py (and flare_classifier.py / Sid scripts if you want those charts):
    python export_charts.py
Writes data/presentation/*.html (one chart each) + plotly.min.js. data/ is git-ignored, so nothing here is
committed. Zip the folder, download it, and add it to the results page.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy.stats import wilcoxon

from precompute import CD8_PANEL, family_of

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = f"{DATA}/presentation"
JUNK = "artifact|Debris|Multiplet|Unknown|Unlabelled|Platelet"
FINE, FLARE, GREY = "#2b7bb9", "#e4572e", "#b0b7c3"
STATE = {"Baseline": "Feeling fine", "Flaring": "Flare"}
FAMILY_COLORS = {"T Cells": "#9b8ce0", "NK Cells": "#3fb8a6", "B Cells": "#8cc63f", "Monocytes": "#f2c14e",
                 "Dendritic Cells": "#e07bb5", "Other": "#bdbdbd"}
STYLE = dict(template="plotly_white", font=dict(family="Inter, Arial, sans-serif", size=13),
             margin=dict(l=60, r=30, t=60, b=50), hoverlabel=dict(font_size=13))


def save(fig, name, height=460):
    fig.update_layout(height=height, **STYLE)
    with open(f"{OUT}/{name}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True}))
    print("  wrote", name)


def per_person_family(fam, samples):
    m = fam.merge(samples[["sample_id", "person", "flare"]], on="sample_id")
    return m.groupby(["family", "person", "flare"])["pct"].mean().unstack("flare").dropna()


def signature(per):
    rows = []
    for f, sub in per.groupby(level=0):
        d = sub["Flaring"] - sub["Baseline"]
        if len(sub) >= 5 and d.abs().sum() > 0:
            rows.append({"family": f, "change": d.mean(), "up": int((d > 0).sum()), "n": len(sub),
                         "p": wilcoxon(sub["Flaring"], sub["Baseline"]).pvalue})
    return pd.DataFrame(rows)


def slope(pairs, title, ylab):
    fig = go.Figure()
    for person, r in pairs.iterrows():
        fig.add_trace(go.Scatter(x=["Feeling fine", "Flare"], y=[r["Baseline"], r["Flaring"]], mode="lines+markers",
                                 line=dict(color=FLARE if r["Flaring"] > r["Baseline"] else FINE, width=2),
                                 opacity=0.75, name=str(person), showlegend=False,
                                 hovertemplate=f"{person}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=["Feeling fine", "Flare"], y=[pairs["Baseline"].mean(), pairs["Flaring"].mean()],
                             mode="lines+markers", line=dict(color="#111", width=5), name="Average"))
    fig.update_layout(title=title, yaxis_title=ylab, xaxis=dict(range=[-0.25, 1.25]))
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plotly.min.js", "w") as f:
        f.write(get_plotlyjs())
    samples = pd.read_parquet(f"{DATA}/samples.parquet")
    samples["state"] = samples["flare"].map(STATE)
    bowel = samples[samples["group"] == "Bowel disease"]
    comp = pd.read_parquet(f"{DATA}/composition.parquet")
    comp = comp[~comp["cell_type"].str.contains(JUNK, case=False)]
    b = comp[comp["level"] == "broad"].copy()
    b["family"] = b["cell_type"].map(family_of)
    fam = b.groupby(["sample_id", "family"], as_index=False)["pct"].sum()

    # 1. study timeline
    t = bowel.sort_values("person")
    fig = px.scatter(t, x="day", y="person", color="state", size="n_cells", size_max=13,
                     color_discrete_map={"Feeling fine": FINE, "Flare": FLARE},
                     hover_data={"disease": True, "n_cells": ":,", "day": ":.0f"},
                     labels={"day": "Days since this person's first sample", "person": "", "state": ""},
                     title="The study: every home blood sample (bowel-disease patients)")
    fig.update_yaxes(categoryorder="category descending")
    save(fig, "01_timeline", 26 * t["person"].nunique() + 140)

    # 2. flare signature by family
    per = per_person_family(fam, bowel)
    sig = signature(per).sort_values("change")
    sig["label"] = sig["up"].astype(str) + "/" + sig["n"].astype(str) + " up · p " + sig["p"].map(lambda p: f"{p:.3f}")
    fig = go.Figure(go.Bar(x=sig["change"], y=sig["family"], orientation="h", text=sig["label"],
                           textposition="outside", cliponaxis=False,
                           marker_color=np.where(sig["p"] < 0.05, np.where(sig["change"] > 0, FLARE, FINE), GREY)))
    pad = sig["change"].abs().max() * 0.6
    fig.update_layout(title="Which immune cell families change during a flare (each person vs themselves)",
                      xaxis=dict(range=[min(sig["change"].min(), 0) - pad, max(sig["change"].max(), 0) + pad],
                                 title="Average change in share of immune cells (percentage points)"),
                      margin=dict(l=140))
    save(fig, "02_signature", 380)

    # 3. B cells person by person
    if "B Cells" in per.index.get_level_values(0):
        save(slope(per.loc["B Cells"], "B cells: each line is one patient", "% of immune cells"), "03_bcells_per_person")

    # 4. steroid robustness for B cells
    if "steroid_hours" in samples:
        rows = []
        for label, s in [("All samples", bowel),
                         ("Without samples within a week of recorded steroids", bowel[~(bowel["steroid_hours"] <= 168)]),
                         ("Without any recorded steroid use", bowel[bowel["steroid_hours"].isna()])]:
            sg = signature(per_person_family(fam, s))
            r = sg[sg["family"] == "B Cells"]
            if len(r):
                r = r.iloc[0]
                rows.append({"label": label, "change": r["change"], "text": f"{r['up']} of {r['n']} up · p {r['p']:.3f}"})
        if rows:
            st = pd.DataFrame(rows)
            fig = go.Figure(go.Bar(x=st["label"], y=st["change"], text=st["text"], textposition="outside",
                                   marker_color=FLARE, cliponaxis=False))
            fig.update_layout(title="B-cell rise with and without recorded steroid use",
                              yaxis=dict(title="Average change (percentage points)",
                                         range=[0, st["change"].max() * 1.35]))
            save(fig, "04_steroids", 400)

    # 5. classifier
    clf_file = f"{DATA}/classifier/results.json"
    if os.path.exists(clf_file):
        clf = json.load(open(clf_file))["results"]
        names = list(clf)
        fig = go.Figure(go.Bar(x=names, y=[clf[n]["correct_ranking"] for n in names],
                               marker_color=[FLARE if clf[n]["shuffle_p"] < 0.05 else GREY for n in names],
                               text=[f"{clf[n]['correct_ranking']:.0%} · {clf[n]['people_flare_higher']}/"
                                     f"{clf[n]['people_with_both']} people · p {clf[n]['shuffle_p']:.3f}" for n in names],
                               textposition="outside", name="Real labels"))
        fig.add_trace(go.Scatter(x=names, y=[clf[n]["shuffle_95th"] for n in names], mode="markers",
                                 name="Shuffled labels: 95% of runs below this",
                                 marker=dict(symbol="line-ew-open", size=40, line=dict(width=3, color="#111"))))
        fig.add_hline(y=0.5, line_dash="dot", line_color=GREY, annotation_text="coin toss", annotation_position="left")
        fig.update_layout(title="Flare classifier, tested on patients it never saw",
                          yaxis=dict(range=[0, 1.1], tickformat=".0%", title="Correct ranking"),
                          legend=dict(orientation="h", y=1.12))
        save(fig, "05_classifier", 440)

    # 6. patient report: P01's highest-B-cell flare sample vs own calm level (or the first person with both)
    w = fam.pivot_table(index="sample_id", columns="family", values="pct", fill_value=0)
    two = bowel.groupby("person")["flare"].nunique().eq(2)
    person = "P01" if two.get("P01", False) else two[two].index[0]
    ps = bowel[bowel["person"] == person]
    calm = w.loc[ps[ps["flare"] == "Baseline"]["sample_id"]].mean()
    flares = ps[ps["flare"] == "Flaring"].copy()
    flares["b"] = w.loc[flares["sample_id"]].get("B Cells", 0).values - calm.get("B Cells", 0)
    pick = flares.sort_values("b").iloc[-1]
    diff = (w.loc[pick["sample_id"]] - calm).sort_values()
    fig = go.Figure(go.Scatter(x=diff.values, y=diff.index, mode="markers",
                               marker=dict(size=18, color=[FAMILY_COLORS.get(f, "#888") for f in diff.index],
                                           line=dict(width=1, color="#333")),
                               hovertemplate="%{y}: %{x:+.1f} points vs own normal<extra></extra>"))
    fig.add_vline(x=0, line_color="#444")
    fig.update_layout(title=f"Patient report: {person}, day {pick['day']:.0f} (flare) vs their own calm samples",
                      xaxis_title="Points above (+) or below (−) their normal share")
    save(fig, "06_report", 380)

    # 7. Crohn's vs ulcerative colitis gene changes in B cells
    pb = pd.read_parquet(f"{DATA}/pseudobulk.parquet")
    genes = [g for g in pb.columns if g not in ("sample_id", "cell_type", "n_cells")]
    bcell_types = [c for c in pb["cell_type"].unique() if family_of(c) == "B Cells"]
    ct = next((c for c in bcell_types if "naive" in c.lower()), bcell_types[0] if bcell_types else None)

    def changes(frame, pattern, cell_type):
        m = frame[frame["cell_type"] == cell_type].merge(
            bowel[bowel["disease"].str.contains(pattern, case=False, regex=False)][["sample_id", "person", "flare"]]
            if pattern else bowel[["sample_id", "person", "flare"]], on="sample_id")
        pp = m.groupby(["person", "flare"])[genes].mean()
        if not {"Baseline", "Flaring"}.issubset(set(pp.index.get_level_values(1))):
            return pd.Series(dtype=float)
        bb, ff = pp.xs("Baseline", level="flare"), pp.xs("Flaring", level="flare")
        ppl = bb.index.intersection(ff.index)
        return (ff.loc[ppl] - bb.loc[ppl]).mean()

    if ct:
        cd, uc = changes(pb, "Crohn", ct), changes(pb, "Ulcerative", ct)
        if len(cd) and len(uc):
            both = pd.DataFrame({"Crohn's": cd, "UC": uc}).dropna()
            fig = px.scatter(both.reset_index(names="gene"), x="Crohn's", y="UC", hover_name="gene", render_mode="webgl",
                             labels={"Crohn's": "Change during flare: Crohn's (log2)", "UC": "Change during flare: UC (log2)"},
                             title=f"Do Crohn's and colitis flares change the same genes? ({ct})")
            fig.update_traces(marker=dict(size=5, color=FLARE, opacity=0.5))
            fig.add_hline(y=0, line_color=GREY)
            fig.add_vline(x=0, line_color=GREY)
            save(fig, "07_crohns_vs_uc", 480)

    # 8. CD8 panel: IBD flares vs Sid around surgery
    sid_pb_file = f"{DATA}/sid/pseudobulk.parquet"
    cd8_types = [c for c in pb["cell_type"].unique() if "cd8" in c.lower()]
    if os.path.exists(sid_pb_file) and cd8_types:
        ibd = changes(pb, None, cd8_types[0])
        sp = pd.read_parquet(sid_pb_file)
        sp["date"] = pd.to_datetime(sp["date"])
        one = sp[sp["cell_type"] == ("CD8 T Cells" if (sp["cell_type"] == "CD8 T Cells").any() else "T Cells")]
        split = pd.Timestamp("2025-04-17")
        sgenes = [g for g in one.columns if g not in ("date", "cell_type", "n_cells")]
        sid = one[one["date"] > split][sgenes].mean() - one[one["date"] < split][sgenes].mean()
        rows = []
        for program, gs in CD8_PANEL.items():
            for g in gs:
                rows.append({"label": f"{program.split(' /')[0]} · {g}", "IBD flare": ibd.get(g, np.nan),
                             "Sid after surgery": sid.get(g, np.nan)})
        panel = pd.DataFrame(rows).dropna(subset=["IBD flare", "Sid after surgery"], how="all")
        z = panel[["IBD flare", "Sid after surgery"]].to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=["IBD flare (CD8, vs own calm)", "Sid (CD8, after vs before surgery)"],
                                   y=panel["label"], zmid=0, colorscale=[[0, FINE], [0.5, "#f7f7f7"], [1, FLARE]],
                                   text=np.where(np.isnan(z), "–", np.round(z, 2).astype(str)), texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="CD8 T-cell inflammation genes: IBD flares vs Sid's cancer (log2 change)",
                          margin=dict(l=200))
        save(fig, "08_cd8_panel", 26 * len(panel) + 140)

    # 9. Sid's T-cell families over time
    clones_file = f"{DATA}/sid/clones.parquet"
    if os.path.exists(clones_file):
        cl = pd.read_parquet(clones_file)
        cl["date"] = pd.to_datetime(cl["date"])
        order = cl.groupby("family")["pct"].max().sort_values(ascending=False).index[:8]
        fig = px.line(cl[cl["family"].isin(order)].sort_values("date"), x="date", y="pct", color="family", markers=True,
                      labels={"date": "", "pct": "% of his T cells", "family": ""},
                      title="Sid's largest T-cell families across his blood draws")
        ev_file = f"{DATA}/sid/events.parquet"
        if os.path.exists(ev_file):
            for _, e in pd.read_parquet(ev_file).dropna().iterrows():
                d = pd.Timestamp(e["date"])
                if cl["date"].min() - pd.Timedelta(days=10) <= d <= cl["date"].max() + pd.Timedelta(days=10):
                    fig.add_vline(x=d.timestamp() * 1000, line_dash="dash", line_color=FLARE,
                                  annotation_text=str(e["label"]).split(" (")[0], annotation_font_color=FLARE)
        save(fig, "09_sid_tcells", 440)

    # 10. cell map by family
    cm = pd.read_parquet(f"{DATA}/cells_map.parquet")
    cm = cm[~cm["broad"].str.contains(JUNK, case=False)]
    cm = cm.sample(min(25_000, len(cm)), random_state=0).assign(family=lambda d: d["broad"].map(family_of))
    fig = px.scatter(cm, x="x", y="y", color="family", color_discrete_map=FAMILY_COLORS, render_mode="webgl",
                     hover_data={"x": False, "y": False, "broad": True}, labels={"family": ""},
                     title="Map of immune cells (each dot one cell, 25,000 shown)")
    fig.update_traces(marker=dict(size=2.5, opacity=0.6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    save(fig, "10_cell_map", 560)
    print(f"Done: {len([f for f in os.listdir(OUT) if f.endswith('.html')])} charts in {OUT}")


if __name__ == "__main__":
    main()
