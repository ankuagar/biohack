"""ImYoo Flare Explorer: what changes in the blood when a bowel-disease flare hits?

Run:  streamlit run app.py
Reads the small tables made by precompute.py (or make_demo_data.py) from ./data.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from precompute import CD8_PANEL, family_of

DATA = os.environ.get("IMYOO_PORTAL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FINE, FLARE = "#2b7bb9", "#e4572e"
GREY = "#b0b7c3"
STATE = {"Baseline": "Feeling fine", "Flaring": "Flare"}
STATE_COLORS = {"Feeling fine": FINE, "Flare": FLARE}
LEVEL_NAMES = {"family": "Cell families", "broad": "Main cell types", "detailed": "Detailed cell types",
               "tcell": "T-cell subsets"}


JUNK = "artifact|Debris|Multiplet|Unknown|Unlabelled|Platelet"
# cell-type colours: deliberately avoid the flare orange and feeling-fine blue
FAMILY_COLORS = {"T Cells": "#9b8ce0", "NK Cells": "#3fb8a6", "B Cells": "#8cc63f", "Monocytes": "#f2c14e",
                 "Dendritic Cells": "#e07bb5", "Proliferating Lymphoid Cells": "#c9a27e",
                 "Myeloid Progenitors": "#8f8f8f", "Lymphoid Progenitors": "#bdbdbd"}
DETAIL_PALETTE = ["#9b8ce0", "#3fb8a6", "#8cc63f", "#f2c14e", "#e07bb5", "#c9a27e", "#6a5acd", "#2e8b57",
                  "#b5d99c", "#d4a017", "#c06c9e", "#7fb3a8", "#a68fcf", "#9acd32", "#e6c27a", "#8f8f8f",
                  "#5f9ea0", "#d8bfd8", "#bdb76b", "#bc8f8f", "#708090", "#cd853f", "#bdbdbd"]

st.set_page_config(page_title="ImYoo Flare Explorer", page_icon="🩸", layout="wide")

PAGES = {"Home": "🏠", "The study": "🗓️", "A person's journey": "👤", "Flare signature": "📊",
         "Seeing a flare coming": "🔮", "Ask the Flare Analyst": "💬", "My immune report": "📋",
         "Sid's journey": "🦴", "Compare conditions": "🔀",
         "Cell map": "🗺️", "Genes": "🧬", "How we checked": "✅"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1320px;}
.hero {background: linear-gradient(135deg, #e4572e 0%, #b8336a 55%, #5b3fd1 100%); color: #fff;
       padding: 2rem 2.3rem; border-radius: 22px; margin-bottom: 1.3rem; box-shadow: 0 12px 32px rgba(184,51,106,.28);}
.hero-kicker {text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; font-weight: 700; opacity: .9;}
.hero-title {font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin: .45rem 0 .7rem;}
.hero-sub {font-size: 1.08rem; opacity: .96; max-width: 62rem; line-height: 1.5;}
.kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .9rem; margin: .3rem 0 1.3rem;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1rem 1.25rem; background: rgba(128,128,128,.06);}
.kpi-icon {font-size: 1.35rem;}
.kpi-val {font-size: 2.05rem; font-weight: 800; line-height: 1.2;}
.kpi-lab {font-size: .86rem; opacity: .72;}
.finding {border-left: 6px solid #e4572e; border-radius: 16px; padding: 1rem 1.35rem; margin: .55rem 0;
          background: rgba(228,87,46,.09);}
.finding.quiet {border-left-color: #b0b7c3; background: rgba(128,128,128,.07);}
.finding .tag {font-size: .74rem; text-transform: uppercase; letter-spacing: .12em; color: #e4572e; font-weight: 800;}
.finding.quiet .tag {color: inherit; opacity: .7;}
.finding .big {font-size: 1.4rem; font-weight: 700; margin: .2rem 0; line-height: 1.3;}
.finding .small {opacity: .75; font-size: .9rem;}
.why-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .9rem; margin: .4rem 0 1rem;}
.why {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 1rem 1.2rem;}
.why b {display: block; font-size: 1.02rem; margin: .3rem 0 .2rem;}
.why span {opacity: .78; font-size: .92rem;}
.page-head {display: flex; align-items: center; gap: .75rem; margin-bottom: .1rem;}
.page-head h1 {margin: 0; padding: 0;}
.page-icon {font-size: 2.3rem; line-height: 1;}
.brand {padding: .2rem 0 .9rem;}
.brand-title {font-size: 1.28rem; font-weight: 800;}
.brand-sub {font-size: .84rem; opacity: .7; margin-top: .15rem;}
.tip {font-size: .88rem; opacity: .75;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: .85rem 1.05rem;
                          background: rgba(128,128,128,.05);}
[data-testid="stExpander"] details {border-radius: 14px;}
[data-testid="stPlotlyChart"] {margin: .25rem 0 .8rem;}
[data-testid="stPlotlyChart"] > div {border-radius: 14px;}
</style>
"""
CSS_PRESENT = """
<style>
html {font-size: 19px;}
.hero-title {font-size: 3rem;}
.finding .big {font-size: 1.65rem;}
</style>
"""


def md(text):
    """Tiny markdown for HTML cards: **bold** and *italic*."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)


def header(title, subtitle=None):
    st.markdown(f"<div class='page-head'><span class='page-icon'>{PAGES.get(page, '')}</span><h1>{title}</h1></div>",
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(subtitle)


def kpis(items):
    """Row of stat cards: items = [(icon, value, label), ...]."""
    cards = "".join(f"<div class='kpi'><div class='kpi-icon'>{i}</div><div class='kpi-val'>{v}</div>"
                    f"<div class='kpi-lab'>{lab}</div></div>" for i, v, lab in items)
    st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)


def finding(tag, big, small="", quiet=False):
    st.markdown(f"<div class='finding{' quiet' if quiet else ''}'><div class='tag'>{tag}</div>"
                f"<div class='big'>{md(big)}</div><div class='small'>{md(small)}</div></div>", unsafe_allow_html=True)


def goto(page_name, **state):
    """Button callback: jump to a page (and pre-fill its pickers)."""
    st.session_state["nav"] = page_name
    st.session_state.update(state)


def chart(fig, where=None, **kw):
    """Apply shared readable styling. Returns Plotly's selection event, when requested."""
    # Transparent backgrounds let charts sit naturally in either Streamlit theme;
    # restrained grid lines and consistent typography make unlike chart types feel related.
    old_margin = fig.layout.margin
    margin = {side: max(default, getattr(old_margin, side, None) or 0)
              for side, default in (("l", 36), ("r", 88), ("t", 72), ("b", 72))}
    margin.update(pad=5, autoexpand=True)
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13, color=avg_color()),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=14, font_family="Inter, sans-serif", bgcolor="#172033",
                        bordercolor="#172033", font=dict(color="#ffffff")),
        margin=margin,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12),
                    title_font=dict(size=12)),
        modebar=dict(orientation="h", bgcolor="rgba(0,0,0,0)", color="#64748b",
                     activecolor=FLARE),
    )
    if getattr(fig.layout.title, "text", None):
        fig.update_layout(title_font=dict(size=16), title_x=0.02, title_xanchor="left")
    fig.update_xaxes(automargin=True, title_standoff=12, showline=False, zerolinecolor="rgba(128,140,160,.35)",
                     gridcolor="rgba(128,140,160,.16)", ticks="outside", ticklen=4,
                     tickcolor="rgba(128,140,160,.35)")
    fig.update_yaxes(automargin=True, title_standoff=12, showline=False, zerolinecolor="rgba(128,140,160,.35)",
                     gridcolor="rgba(128,140,160,.16)", ticks="outside", ticklen=4,
                     tickcolor="rgba(128,140,160,.35)")
    return (where or st).plotly_chart(fig, width="stretch", config={"displaylogo": False}, **kw)


def sample_labels(s):
    """'Day 12 · Flare' per sample of one person (two aliquots on the same day get '(2)')."""
    labels, seen = {}, {}
    for _, r in s.sort_values("day").iterrows():
        lab = f"Day {r['day']:.0f} · {r['state']}"
        seen[lab] = seen.get(lab, 0) + 1
        labels[r["sample_id"]] = lab if seen[lab] == 1 else f"{lab} ({seen[lab]})"
    return labels


# ---------- data ----------

@st.cache_resource
def _last_build():
    return {"built_at": None}


def load():
    """ImYoo summaries, re-read automatically whenever precompute.py rebuilds them.

    A rebuild also wipes every remembered result (scores, comparisons) so nothing stale is shown.
    """
    built_at = os.path.getmtime(f"{DATA}/info.json")
    seen = _last_build()
    if seen["built_at"] not in (None, built_at):
        st.cache_data.clear()
    seen["built_at"] = built_at
    return _load(built_at)


@st.cache_data
def _load(built_at):
    d = {name: pd.read_parquet(f"{DATA}/{name}.parquet")
         for name in ["samples", "composition", "cells_map", "pseudobulk", "gene_tests"]}
    with open(f"{DATA}/info.json") as f:
        d["info"] = json.load(f)
    d["samples"]["state"] = d["samples"]["flare"].map(STATE)
    c = d["composition"]
    d["composition"] = c[~c["cell_type"].str.contains(JUNK, case=False)]
    m = d["cells_map"]
    d["cells_map"] = m[~m["broad"].str.contains(JUNK, case=False) & ~m["detailed"].str.contains(JUNK, case=False)]
    d["cells_map"]["family"] = d["cells_map"]["broad"].map(family_of)
    # Families: add subtypes back together, only when the main level is split finer than families (v3)
    b = d["composition"][d["composition"]["level"] == "broad"].copy()
    b["cell_type"] = b["cell_type"].map(family_of)
    if b["cell_type"].nunique() < d["composition"].loc[d["composition"]["level"] == "broad", "cell_type"].nunique():
        fam = b.groupby(["sample_id", "cell_type"], as_index=False)["pct"].sum().assign(level="family")
        d["composition"] = pd.concat([d["composition"], fam], ignore_index=True)
    return d


def load_sid():
    """Sid Sijbrandij's blood summaries from sid_precompute.py, or None if not built yet.

    Checked fresh on every visit (not remembered), and re-read whenever the summaries are rebuilt.
    """
    info = f"{DATA}/sid/info.json"
    return _read_sid(f"{DATA}/sid", os.path.getmtime(info)) if os.path.exists(info) else None


@st.cache_data
def _read_sid(folder, built_at):
    d = {n: pd.read_parquet(f"{folder}/{n}.parquet")
         for n in ["draws", "composition", "cells_map", "clones", "tcell_diversity", "markers", "events"]}
    d["groups"] = pd.read_parquet(f"{folder}/groups.parquet") if os.path.exists(f"{folder}/groups.parquet") else None
    for extra in ["pseudobulk", "gene_names"]:  # made by sid_pseudobulk.py (optional)
        f = f"{folder}/{extra}.parquet"
        d[extra] = pd.read_parquet(f) if os.path.exists(f) else None
    if d["pseudobulk"] is not None:
        d["pseudobulk"]["date"] = pd.to_datetime(d["pseudobulk"]["date"])
    with open(f"{folder}/info.json") as f:
        d["info"] = json.load(f)
    for n in ["draws", "composition", "cells_map", "clones", "tcell_diversity"]:
        d[n]["date"] = pd.to_datetime(d[n]["date"])
    d["events"] = d["events"].dropna()
    d["events"]["date"] = pd.to_datetime(d["events"]["date"])
    return d


def mark_events(fig, events, start, end):
    """Dashed line + label for each tumour sampling date that falls inside the chart's time range."""
    for _, e in events.iterrows():
        if start - pd.Timedelta(days=10) <= e["date"] <= end + pd.Timedelta(days=10):
            fig.add_shape(type="line", x0=e["date"], x1=e["date"], y0=0, y1=1, yref="paper",
                          line=dict(color=FLARE, dash="dash", width=2))
            fig.add_annotation(x=e["date"], y=1.02, yref="paper", text=e["label"].split(" (")[0] + " " +
                               e["label"].split("(")[-1].rstrip(")"), showarrow=False,
                               font=dict(color=FLARE, size=12), xanchor="left")


def type_colors(level):
    return FAMILY_COLORS if level in ("family", "broad") else None


STEROID_WINDOWS = {"None": None, "1 day": 24, "3 days": 72, "1 week": 168, "1 month": 720,
                   "Any recorded steroid use": float("inf")}
STEROID_CAVEAT = ("Steroid timing was extracted by an AI from patients' answers, without human checking. A recorded "
                  "time is probably right, but a blank can mean no steroids, not reported, or missed, so some "
                  "steroid use may be unrecorded.")


def paired_cell_stats(level, group, drop_runs, same_batch=False, disease=None, steroid_hours=None):
    """For each cell type: each person's flare share vs their own baseline share, tested across people.

    steroid_hours: leave out samples taken within this many hours of RECORDED steroid use (blanks are kept: they
    may be no steroids, or unreported/missed steroids, see STEROID_CAVEAT).
    """
    return _paired_cell_stats(level, group, drop_runs, same_batch, disease, steroid_hours, INFO.get("built"))


@st.cache_data
def _paired_cell_stats(level, group, drop_runs, same_batch, disease, steroid_hours, built):
    s = D["samples"]
    if steroid_hours and "steroid_hours" in s:
        s = s[~(s["steroid_hours"] <= steroid_hours)]
    if group != "Everyone":
        s = s[s["group"] == group]
    if disease:
        s = s[s["disease"].str.contains(disease, case=False, regex=False)]
    s = s[~s["run"].isin(drop_runs)]
    if same_batch:
        one = s.groupby("person")["run"].nunique().eq(1)
        s = s[s["person"].isin(one[one].index)]
    c = D["composition"]
    c = c[c["level"] == level].merge(s[["sample_id", "person", "flare"]], on="sample_id")
    per = c.groupby(["cell_type", "person", "flare"])["pct"].mean().unstack("flare").dropna()
    rows, pairs = [], []
    for ct, sub in per.groupby(level=0):
        b, f = sub["Baseline"], sub["Flaring"]
        if len(sub) < 5 or (f - b).abs().sum() == 0:
            continue
        rows.append({"cell type": ct, "fine %": b.mean(), "flare %": f.mean(),
                     "change": (f - b).mean(), "people up": int((f > b).sum()),
                     "people": len(sub), "p": wilcoxon(f, b).pvalue})
        pairs.append(sub.reset_index())
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    res = pd.DataFrame(rows)
    res["adjusted p"] = multipletests(res["p"], method="fdr_bh")[1]
    res = res.sort_values("p").reset_index(drop=True)
    return res, pd.concat(pairs, ignore_index=True)


@st.cache_data
def flare_scores(group, level="detailed"):
    """Score every sample for how flare-like its immune-cell mix looks.

    Each person's scores come from a model trained WITHOUT that person (leave-one-person-out),
    so the score can't simply memorise people.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    s = D["samples"] if group == "Everyone" else D["samples"][D["samples"]["group"] == group]
    s = s.set_index("sample_id")
    c = D["composition"]
    X = (c[c["level"] == level].pivot_table(index="sample_id", columns="cell_type", values="pct", fill_value=0)
         .reindex(s.index).fillna(0))
    X = np.log(X + 0.1)
    y = (s["flare"] == "Flaring").astype(int).to_numpy()
    persons = s["person"].to_numpy()

    def fit(rows):
        scaler = StandardScaler().fit(X.values[rows])
        model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
        return scaler, model.fit(scaler.transform(X.values[rows]), y[rows])

    score = np.full(len(X), np.nan)
    for p in np.unique(persons):
        train = persons != p
        if y[train].min() == y[train].max():
            continue
        scaler, model = fit(train)
        score[~train] = model.predict_proba(scaler.transform(X.values[~train]))[:, 1]
    _, model = fit(np.ones(len(X), dtype=bool))
    weights = pd.Series(model.coef_[0], index=X.columns).sort_values()
    out = s.reset_index()
    out["score"] = score
    ok = ~np.isnan(score)
    auc = roc_auc_score(y[ok], score[ok]) if len(set(y[ok])) == 2 else float("nan")
    return out, weights, auc


@st.cache_data
def personal_scores(group, kind):
    """Compare each sample with the SAME person's OTHER calm samples (never with itself).

    kind = "bcell": change in B-cell share (percentage points), no model at all.
    kind = "model": every cell type's change from the person's own calm level, scored by a model
                    trained without that person (leave-one-person-out).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    s = D["samples"] if group == "Everyone" else D["samples"][D["samples"]["group"] == group]
    s = s.set_index("sample_id")
    c = D["composition"]
    level = FAM if kind == "bcell" else "detailed"
    X = (c[c["level"] == level].pivot_table(index="sample_id", columns="cell_type", values="pct", fill_value=0)
         .reindex(s.index).fillna(0))
    if kind == "model":
        X = np.log(X + 0.1)
    calm = (s["flare"] == "Baseline").to_numpy()
    persons = s["person"].to_numpy()
    delta = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    for p in np.unique(persons):
        idx = np.where(persons == p)[0]
        for i in idx:
            others = [j for j in idx if j != i and calm[j]]
            if others:
                delta.iloc[i] = X.values[i] - X.values[others].mean(axis=0)
    y = (s["flare"] == "Flaring").astype(int).to_numpy()
    has_ref = delta.notna().all(axis=1).to_numpy()
    weights = None
    if kind == "bcell":
        score = delta["B Cells"].to_numpy() if "B Cells" in delta else np.full(len(X), np.nan)
    else:
        score = np.full(len(X), np.nan)
        for p in np.unique(persons[has_ref]):
            train = has_ref & (persons != p)
            test = has_ref & (persons == p)
            if y[train].min() == y[train].max():
                continue
            scaler = StandardScaler().fit(delta.values[train])
            model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
            model.fit(scaler.transform(delta.values[train]), y[train])
            score[test] = model.predict_proba(scaler.transform(delta.values[test]))[:, 1]
        if has_ref.sum() and len(set(y[has_ref])) == 2:
            scaler = StandardScaler().fit(delta.values[has_ref])
            model = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
            model.fit(scaler.transform(delta.values[has_ref]), y[has_ref])
            weights = pd.Series(model.coef_[0], index=X.columns).sort_values()
    out = s.reset_index()
    out["score"] = score
    ok = ~np.isnan(score)
    auc = roc_auc_score(y[ok], score[ok]) if ok.sum() and len(set(y[ok])) == 2 else float("nan")
    return out, weights, auc


@st.cache_data
def personal_family_deltas():
    """Each sample vs the same person's OTHER calm samples, for every broad cell family (percentage points).

    Also returns each person's normal (mean of their calm samples) and every calm sample's change from its own
    normal (used to say how unusual a change is: "higher than X% of calm samples in the study").
    """
    s = D["samples"].set_index("sample_id")
    c = D["composition"]
    X = (c[c["level"] == FAM].pivot_table(index="sample_id", columns="cell_type", values="pct", fill_value=0)
         .reindex(s.index).fillna(0))
    calm = (s["flare"] == "Baseline").to_numpy()
    persons = s["person"].to_numpy()
    delta = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    normal = {}
    for p in np.unique(persons):
        idx = np.where(persons == p)[0]
        calm_idx = [j for j in idx if calm[j]]
        if calm_idx:
            normal[p] = X.iloc[calm_idx].mean()
        for i in idx:
            others = [j for j in calm_idx if j != i]
            if others:
                delta.iloc[i] = X.values[i] - X.values[others].mean(axis=0)
    return X, delta, pd.DataFrame(normal).T, delta[calm]


DISEASES = [("Crohn's", "Crohn"), ("Ulcerative colitis", "Ulcerative")]


@st.cache_data
def flare_gene_changes(cell_type, disease_pattern=None):
    """Per gene: average change during a flare, each person vs their own calm samples (log2 units).

    disease_pattern: text the disease label must contain (e.g. "Crohn"), or None for all bowel-disease people.
    """
    pb = D["pseudobulk"]
    pb = pb[pb["cell_type"] == cell_type]
    genes = [c for c in pb.columns if c not in ("sample_id", "cell_type", "n_cells")]
    s = D["samples"][D["samples"]["group"] == "Bowel disease"]
    if disease_pattern:
        s = s[s["disease"].str.contains(disease_pattern, case=False, regex=False)]
    m = pb.merge(s[["sample_id", "person", "flare"]], on="sample_id")
    per = m.groupby(["person", "flare"])[genes].mean()
    if per.empty or "Baseline" not in per.index.get_level_values(1) or "Flaring" not in per.index.get_level_values(1):
        return pd.DataFrame()
    base, fl = per.xs("Baseline", level="flare"), per.xs("Flaring", level="flare")
    people = base.index.intersection(fl.index)
    if len(people) < 3:
        return pd.DataFrame()
    diff = fl.loc[people] - base.loc[people]
    return pd.DataFrame({"change": diff.mean(), "people_up": (diff > 0).sum(), "people": len(people),
                         "activity": ((fl.loc[people] + base.loc[people]) / 2).mean()})


def classify(a, b, size=0.5):
    """Plain-language agreement between two gene changes (log2 units; 0.5 = about 1.4-fold)."""
    up_a, up_b, dn_a, dn_b = a > size, b > size, a < -size, b < -size
    return np.select([up_a & up_b, dn_a & dn_b, (up_a & dn_b) | (dn_a & up_b), up_a | dn_a, up_b | dn_b],
                     ["Up in both", "Down in both", "Opposite directions", "First only", "Second only"],
                     default="Little change")


COMPARE_COLORS = {"Up in both": FLARE, "Down in both": FINE, "Opposite directions": "#8e44ad",
                  "First only": "#f2c14e", "Second only": "#3fb8a6", "Little change": GREY}


def compare_scatter(df, xname, yname, xlab, ylab, size=0.5):
    """Scatter of two sets of gene changes. Similarity is measured on genes that clearly change in at least one
    condition (genes that barely move only add noise); the all-genes value is returned too."""
    from scipy.stats import spearmanr
    moving = df[(df["x"].abs() > size) | (df["y"].abs() > size)]
    rho_all = spearmanr(df["x"], df["y"]).correlation if len(df) > 3 else float("nan")
    use_moving = len(moving) >= 8
    rho = spearmanr(moving["x"], moving["y"]).correlation if use_moving else rho_all
    df = df.assign(pattern=classify(df["x"], df["y"], size))
    df.attrs.update(rho_all=rho_all, n_moving=len(moving),
                    basis=f"the {len(moving)} genes that clearly change in at least one" if use_moving
                    else f"all {len(df)} genes (too few change clearly)")
    df["pattern"] = df["pattern"].replace({"First only": f"{xname} only", "Second only": f"{yname} only"})
    colors = {**COMPARE_COLORS, f"{xname} only": COMPARE_COLORS["First only"],
              f"{yname} only": COMPARE_COLORS["Second only"]}
    fig = px.scatter(df.reset_index(), x="x", y="y", color="pattern", hover_name="gene", render_mode="webgl",
                     color_discrete_map=colors, labels={"x": xlab, "y": ylab, "pattern": ""})
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    fig.update_layout(title=f"Gene activity change: {xname} compared with {yname}", height=520)
    return fig, rho, df


def similarity_words(rho):
    if rho != rho:
        return "not enough genes to compare"
    if rho > 0.5:
        return "strongly similar"
    if rho > 0.2:
        return "somewhat similar"
    if rho > -0.2:
        return "largely unrelated"
    return "opposite patterns"


def countdown(scored):
    """For each feeling-fine sample: days until that person's next flare sample (if any was seen)."""
    rows = []
    for _, g in scored.groupby("person"):
        flares = g.loc[g["flare"] == "Flaring", "day"].to_numpy()
        last_day = g["day"].max()
        for _, r in g[g["flare"] == "Baseline"].iterrows():
            later = flares[flares > r["day"]]
            rows.append({**r.to_dict(),
                         "days_to_flare": later.min() - r["day"] if len(later) else np.nan,
                         "is_last": r["day"] == last_day})
    return pd.DataFrame(rows)


def signature_bar(res):
    """Bar per cell type: average change during flare, coloured if it passes the strict test."""
    res = res.copy()
    res["direction"] = np.where(res["adjusted p"] < 0.05,
                                np.where(res["change"] > 0, "Up in flare", "Down in flare"), "No clear change")
    res["label"] = res["people up"].astype(str) + "/" + res["people"].astype(str) + " up"
    fig = px.bar(res.sort_values("change"), x="change", y="cell type", orientation="h", color="direction", text="label",
                 color_discrete_map={"Up in flare": FLARE, "Down in flare": FINE, "No clear change": GREY},
                 hover_data={"people up": True, "people": True, "adjusted p": ":.3f"},
                 labels={"change": "Average change in immune cell share (percentage points)",
                         "cell type": "Cell type"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="How each cell type changes during a flare",
                      height=max(350, 28 * len(res)), legend_title_text="")
    return fig


def verdict(row):
    direction = "rise" if row["change"] > 0 else "fall"
    consistent = row["people up"] if row["change"] > 0 else row["people"] - row["people up"]
    return direction, consistent


def avg_color():
    """White on a dark page, black on a light one, so the average line is always visible."""
    try:
        return "#ffffff" if st.context.theme.type == "dark" else "#111111"
    except Exception:
        return "#888888"


def slope_chart(pairs, value_label, title):
    """One line per person from 'Feeling fine' to 'Flare'. The picture judges can count."""
    fig = go.Figure()
    for _, r in pairs.iterrows():
        up = r["Flaring"] > r["Baseline"]
        fig.add_trace(go.Scatter(
            x=["Feeling fine", "Flare"], y=[r["Baseline"], r["Flaring"]], mode="lines+markers",
            line=dict(color=FLARE if up else FINE, width=1.4), marker=dict(size=5), opacity=0.42,
            name=r["person"], hovertemplate=f"{r['person']}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
            showlegend=False))
    # Per-person traces are intentionally hidden from the legend to avoid a long
    # list of IDs. Add a compact key that explains their colour instead.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", name="Person: higher in flare",
                             line=dict(color=FLARE, width=3), marker=dict(size=7),
                             hoverinfo="skip", legendrank=1))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", name="Person: lower in flare",
                             line=dict(color=FINE, width=3), marker=dict(size=7),
                             hoverinfo="skip", legendrank=2))
    fig.add_trace(go.Scatter(
        x=["Feeling fine", "Flare"], y=[pairs["Baseline"].mean(), pairs["Flaring"].mean()],
        mode="lines+markers", line=dict(color=avg_color(), width=5), marker=dict(size=12),
        name="Study average", hovertemplate="Study average %{x}: %{y:.2f}<extra></extra>", legendrank=3))
    fig.update_layout(title=dict(text=title, x=0.02, y=0.98, xanchor="left", yanchor="top"),
                      yaxis_title=value_label, height=500,
                      margin=dict(t=92, b=78, l=72, r=250),
                      xaxis=dict(range=[-0.25, 1.25], title="Blood sample condition"),
                      legend=dict(title="Lines", orientation="v", y=0.98, x=1.02,
                                  xanchor="left", yanchor="top"))
    return fig


def pval_text(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def analyst_tools():
    """The analyses Claude may run. Each returns (small summary dict, chart or None)."""
    from scipy.stats import mannwhitneyu

    def cols(res):
        return res[["cell type", "fine %", "flare %", "change", "people up", "people", "adjusted p"]].to_dict("records")

    def list_options():
        c = D["composition"]
        return ({"cell_types": {lv: sorted(c.loc[c["level"] == lv, "cell_type"].unique()) for lv in LEVELS},
                 "gene_cell_types": sorted(D["gene_tests"]["cell_type"].unique()),
                 "batches": sorted(D["samples"]["run"].unique()),
                 "diseases": sorted(D["samples"]["disease"].unique())}, None)

    def study_overview(group):
        s = D["samples"] if group == "Everyone" else D["samples"][D["samples"]["group"] == group]
        return ({"people": s["person"].nunique(), "samples": len(s),
                 "flare_samples": int((s["flare"] == "Flaring").sum()),
                 "calm_samples": int((s["flare"] == "Baseline").sum()),
                 "people_with_both": int(s.groupby("person")["flare"].nunique().eq(2).sum()),
                 "flare_samples_within_week_of_steroids": int(((s["flare"] == "Flaring") & (s.get("steroid_hours", pd.Series(dtype=float)) <= 168)).sum()) if "steroid_hours" in s else None,
                 "median_age": float(s.groupby("person")["age"].first().median()) if "age" in s else None,
                 "conditions": s.groupby("disease")["person"].nunique().to_dict()}, None)

    def compare_cell_types(level, group, leave_out_batch=None, same_batch_only=False, disease=None,
                           exclude_within_hours_of_steroids=None):
        drop = (leave_out_batch,) if leave_out_batch else ()
        res, _ = paired_cell_stats(level, group, drop, bool(same_batch_only), disease,
                                   exclude_within_hours_of_steroids)
        if res.empty:
            return {"note": "Not enough people with both flare and calm samples for this selection."}, None
        passing = res[res["adjusted p"] < 0.05]["cell type"].tolist()
        return {"people_compared": int(res["people"].max()), "passes_strict_test": passing,
                "cell_types": cols(res)}, signature_bar(res)

    def cell_type_person_by_person(level, cell_type, group):
        res, pairs = paired_cell_stats(level, group, ())
        sub = pairs[pairs["cell_type"] == cell_type]
        if sub.empty:
            return {"error": f"Unknown cell type '{cell_type}' at level '{level}'. Use list_options."}, None
        row = res[res["cell type"] == cell_type].iloc[0]
        changes = (sub["Flaring"] - sub["Baseline"]).round(2)
        return ({"people": len(sub), "people_up": int((changes > 0).sum()),
                 "average_fine_pct": sub["Baseline"].mean(), "average_flare_pct": sub["Flaring"].mean(),
                 "biggest_rise": changes.max(), "biggest_fall": changes.min(), "adjusted_p": row["adjusted p"]},
                slope_chart(sub, "% of immune cells", cell_type))

    def gene_changes(cell_type, direction="both", top_n=10):
        g = D["gene_tests"]
        g = g[g["cell_type"] == cell_type]
        if g.empty:
            return {"error": f"No gene results for '{cell_type}'. Use list_options (gene_cell_types)."}, None
        if direction == "up":
            g = g[g["change"] > 0]
        elif direction == "down":
            g = g[g["change"] < 0]
        top = g.sort_values("p").head(int(top_n))
        fig = px.bar(top.sort_values("change"), x="change", y="gene", orientation="h",
                     color=np.where(top.sort_values("change")["change"] > 0, "More active in flare", "Less active in flare"),
                     color_discrete_map={"More active in flare": FLARE, "Less active in flare": FINE},
                     labels={"change": "Change in gene activity during flare (log2)", "gene": "Gene", "color": ""})
        fig.update_layout(title="Genes with the largest flare related activity changes")
        fig.update_layout(height=max(300, 26 * len(top)), legend_title_text="")
        return ({"genes_passing_adjusted_p_0.1": int((g["padj"] < 0.1).sum()), "genes_tested": len(g),
                 "top_genes": top[["gene", "change", "people_up", "people", "padj"]].to_dict("records")}, fig)

    def flare_score(kind, group, window_days=45):
        if kind == "general":
            scored, _, auc = flare_scores(group)
        else:
            scored, _, auc = personal_scores(group, "bcell" if kind == "personal_bcell" else "model")
        cd = countdown(scored).dropna(subset=["score"])
        soon = cd[cd["days_to_flare"] <= window_days]["score"]
        rest = cd[(cd["days_to_flare"].isna() | (cd["days_to_flare"] > window_days)) & ~cd["is_last"]]["score"]
        out = {"correct_ranking": auc, "calm_samples_with_flare_soon": len(soon), "other_calm_samples": len(rest)}
        if len(soon) >= 3 and len(rest) >= 3:
            out.update({"median_score_flare_soon": soon.median(), "median_score_rest": rest.median(),
                        "p_flare_soon_higher": mannwhitneyu(soon, rest, alternative="greater").pvalue})
        fig = px.strip(scored.dropna(subset=["score"]), x="state", y="score", color="state",
                       color_discrete_map=STATE_COLORS, labels={"state": "", "score": "Score"})
        fig.update_layout(title="Flare score in flare and calm samples", height=300, showlegend=False,
                          xaxis_title="Sample status")
        return out, fig

    def red_flag(limit, group):
        scored, _, _ = personal_scores(group, "bcell")
        d = scored.dropna(subset=["score"])
        fl, ca = d[d["flare"] == "Flaring"], d[d["flare"] == "Baseline"]
        hit = fl[fl["score"] > limit]
        return ({"limit_points": limit, "flares_flagged": len(hit), "flares_total": len(fl),
                 "calm_flagged_false_alarms": int((ca["score"] > limit).sum()), "calm_total": len(ca),
                 "different_people_flagged": hit["person"].nunique(),
                 "note": "threshold idea came from looking at this same data; needs confirming"}, None)

    return {f.__name__: f for f in [list_options, study_overview, compare_cell_types, cell_type_person_by_person,
                                    gene_changes, flare_score, red_flag]}


D = load()
INFO = D["info"]
LEVELS = [k for k in LEVEL_NAMES if (D["composition"]["level"] == k).any()]  # detail levels this data has
FAM = "family" if "family" in LEVELS else "broad"  # the level where B cells, T cells... are whole families

# ---------- sidebar ----------

st.sidebar.markdown("<div class='brand'><div class='brand-title'>🩸 ImYoo Flare Explorer</div>"
                    "<div class='brand-sub'>What changes in the blood when a flare hits?</div></div>",
                    unsafe_allow_html=True)
page = st.sidebar.radio("Explore", list(PAGES), key="nav", format_func=lambda p: f"{PAGES[p]}  {p}",
                        label_visibility="collapsed")
st.sidebar.divider()
group = st.sidebar.selectbox("Who to include", ["Bowel disease", "Everyone", "Other"],
                             help="'Other' = the people with arthritis, lupus or fibromyalgia instead of bowel disease.")
present = st.sidebar.toggle("🎤 Presentation mode", help="Bigger text for the big screen.")
st.sidebar.caption(f"People are shown as P01–P{INFO['people']:02d}. Data stays inside SPC's AWS. "
                   "No attempt is made to identify anyone.")
st.sidebar.caption("Built at the SPC Bio-Data Hackathon · data from ImYoo")
st.markdown(CSS + (CSS_PRESENT if present else ""), unsafe_allow_html=True)
if INFO.get("demo"):
    st.warning("**DEMO MODE: made-up numbers.** Run precompute.py inside SageMaker to see the real results.")

samples_g = D["samples"] if group == "Everyone" else D["samples"][D["samples"]["group"] == group]


# ---------- pages ----------

if page == "Home":
    st.markdown("<div class='hero'><div class='hero-kicker'>SPC Bio-Data Hackathon · ImYoo at-home blood study</div>"
                "<div class='hero-title'>What changes in the blood when a flare hits?</div>"
                "<div class='hero-sub'>People with Crohn's disease and ulcerative colitis live with <b>flares</b>: "
                "weeks when symptoms suddenly get much worse. Patients collected blood <b>at home</b>, during flares "
                "and when they felt fine, and we looked at it cell by cell, comparing each person with "
                "<b>their own normal</b>.</div></div>", unsafe_allow_html=True)
    kpis([("👥", INFO["people"], "people"), ("🩸", INFO["samples"], "home blood samples"),
          ("🔬", f"{INFO['cells_kept']:,}", "immune cells"), ("🧬", f"{INFO['genes_total']:,}", "genes measured")])

    who = {"Bowel disease": "bowel-disease patients", "Everyone": "everyone", "Other": "people with other conditions"}[group]
    st.subheader(f"What we found ({who})")
    fam_res, _ = paired_cell_stats(FAM, group, ())
    hyp = fam_res[fam_res["cell type"] == "B Cells"] if len(fam_res) else fam_res
    if len(hyp):
        h = hyp.iloc[0]
        direction, consistent = verdict(h)
        finding("Our hypothesis, re-tested on the newest data",
                f"During a flare, **B Cells** {direction} in **{consistent} of {h['people']}** people",
                f"Average change {h['change']:+.1f} percentage points. Tested on its own (we chose this before the "
                f"newest data arrived): p {pval_text(h['p'])}. Corrected for testing all {len(fam_res)} cell families: "
                f"adjusted p {pval_text(h['adjusted p'])}.",
                quiet=h["p"] >= 0.05)
    shown = {"B Cells"}
    for level, intro in [(FAM, "Big picture"), ("detailed", "Zooming in")]:
        res, _ = paired_cell_stats(level, group, ())
        for _, top in res[res["adjusted p"] < 0.05].head(2).iterrows():
            if top["cell type"] in shown:
                continue
            shown.add(top["cell type"])
            direction, consistent = verdict(top)
            finding(intro, f"During a flare, **{top['cell type']}** {direction} in **{consistent} of {top['people']}** "
                    "people", f"Average change {top['change']:+.1f} percentage points · adjusted p "
                    f"{pval_text(top['adjusted p'])} · each person compared with their own calm samples")
    if len(shown) == 1 and len(fam_res):
        lead = fam_res[fam_res["cell type"] != "B Cells"].head(1)
        if len(lead):
            top = lead.iloc[0]
            direction, consistent = verdict(top)
            finding("Other cell families", f"**{top['cell type']}** {direction} in {consistent} of {top['people']} "
                    "people", f"adjusted p {pval_text(top['adjusted p'])}: a trend, doesn't pass the strict test",
                    quiet=True)

    st.subheader("Take the tour")
    tour = [("📊 The finding", "Flare signature"), ("📋 A patient's report", "My immune report"),
            ("💬 Ask the AI analyst", "Ask the Flare Analyst"), ("✅ Is it real?", "How we checked")]
    for col, (label, target) in zip(st.columns(len(tour)), tour):
        col.button(label, on_click=goto, args=(target,), width="stretch", type="primary" if target == "Flare signature" else "secondary")

    st.subheader("Why this comparison is convincing")
    st.markdown("<div class='why-row'>"
                "<div class='why'>🪞<b>Everyone vs themselves</b><span>Flare blood against their own calm blood, so "
                "differences between people don't muddy the picture.</span></div>"
                "<div class='why'>🏠<b>Real life, not a clinic</b><span>Samples taken at home, on the days flares "
                "actually happened.</span></div>"
                "<div class='why'>🛡️<b>Checked for the usual traps</b><span>Lab batches, freezer time, junk cells, "
                "testing many things at once.</span></div></div>", unsafe_allow_html=True)

elif page == "The study":
    header("The study at a glance")
    st.markdown("Each row is one person. Each dot is a blood sample they collected at home. "
                f"**<span style='color:{FLARE}'>Orange = flare</span>**, "
                f"**<span style='color:{FINE}'>blue = feeling fine</span>**.", unsafe_allow_html=True)
    s = samples_g.sort_values("person")
    fig = px.scatter(s, x="day", y="person", color="state", color_discrete_map=STATE_COLORS, custom_data=["sample_id"],
                     size="n_cells", size_max=16,
                     hover_data={"disease": True, "n_cells": ":,", "day": ":.0f",
                                 **({"age": ":.0f"} if "age" in s else {}),
                                 **({"steroid_hours": ":.0f"} if "steroid_hours" in s else {})},
                     labels={"day": "Days since each person's first sample", "person": "Participant", "state": ""})
    fig.update_yaxes(categoryorder="category descending")
    fig.update_layout(title="When each participant gave a blood sample",
                      height=max(400, 22 * s["person"].nunique()), legend=dict(orientation="h", y=1.05),
                      clickmode="event+select")
    event = chart(fig, on_select="rerun", selection_mode="points", key="study_pick")
    picked = [p for p in (event.selection.points if event else []) if p.get("customdata")]
    if picked:
        sid = picked[0]["customdata"][0]
        r = s[s["sample_id"] == sid].iloc[0]
        lab = sample_labels(D["samples"][D["samples"]["person"] == r["person"]])[sid]
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"**{r['person']} · {lab}** · {r['disease']} · {int(r['n_cells']):,} cells")
        c2.button(f"📋 Open {r['person']}'s immune report", on_click=goto, type="primary", width="stretch",
                  args=("My immune report",), kwargs={"report_person": r["person"], "report_sample": lab})
    else:
        st.markdown("<div class='tip'>👆 Click any dot to open that blood sample's immune report.</div>",
                    unsafe_allow_html=True)
    c = st.columns(3)
    c[0].metric("Flare samples", int((s["flare"] == "Flaring").sum()))
    c[1].metric("Feeling-fine samples", int((s["flare"] == "Baseline").sum()))
    both = s.groupby("person")["flare"].nunique().eq(2).sum()
    c[2].metric("People with both", f"{both} of {s['person'].nunique()}")
    with st.expander("Which conditions do people have?"):
        st.dataframe(s.groupby("disease")["person"].nunique().sort_values(ascending=False)
                     .rename("people").reset_index(), hide_index=True)

elif page == "A person's journey":
    header("One person's immune system over time")
    people = sorted(samples_g["person"].unique())
    multi = samples_g.groupby("person")["flare"].nunique().eq(2)
    default = next((p for p in people if multi.get(p, False)), people[0])
    person = st.selectbox("Pick a person", people, index=people.index(default))
    level = st.radio("Detail", LEVELS, format_func=LEVEL_NAMES.get, horizontal=True)
    s = samples_g[samples_g["person"] == person].sort_values("day")
    st.caption(f"Condition: {s['disease'].iloc[0]} · {len(s)} samples over {s['day'].max() - s['day'].min():.0f} days")
    c = D["composition"]
    c = c[(c["level"] == level) & c["sample_id"].isin(s["sample_id"])].merge(
        s[["sample_id", "day", "state"]], on="sample_id")
    c["label"] = c["day"].map(lambda d: f"Day {d:.0f}") + " · " + c["state"]
    order = (s["day"].map(lambda d: f"Day {d:.0f}") + " · " + s["state"]).tolist()
    fig = px.bar(c, x="label", y="pct", color="cell_type", category_orders={"label": order},
                 color_discrete_map=type_colors(level), color_discrete_sequence=DETAIL_PALETTE,
                 labels={"label": "Sample date and status", "pct": "Immune cells (%)", "cell_type": "Cell type"})
    for i, st_ in enumerate(s["state"]):
        if st_ == "Flare":
            fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor=FLARE, opacity=0.25, line_width=0, layer="below",
                          annotation_text="Flare", annotation_position="top",
                          annotation_font_color=FLARE)
    fig.update_layout(title="Immune cell mix in each blood sample", height=480, barmode="stack",
                      xaxis_title="Sample date and flare status", yaxis_title="Share of immune cells (%)")
    chart(fig)
    st.caption("Shaded columns are flares. Each bar is that day's mix of immune cells.")
    if s["flare"].nunique() == 2:
        own = c.groupby(["cell_type", "state"])["pct"].mean().unstack("state")
        own["change during flare"] = own["Flare"] - own["Feeling fine"]
        own = own.sort_values("change during flare", key=np.abs, ascending=False).round(2)
        st.subheader(f"{person}: what's different during their flares")
        st.dataframe(own, width="stretch")
    else:
        st.info("This person only has one kind of sample, so we can't compare flare vs feeling fine for them.")

elif page == "Flare signature":
    header("Which immune cells change during a flare?")
    st.markdown("For each person who gave **both** kinds of sample, we compare their flare blood with their own "
                "feeling-fine blood, then ask: **does the change point the same way for most people?**")
    c1, c2 = st.columns([2, 1])
    level = c1.radio("Detail", LEVELS, format_func=LEVEL_NAMES.get, horizontal=True,
                     index=LEVELS.index(FAM) if FAM == "family" else min(1, len(LEVELS) - 1))
    runs = sorted(D["samples"]["run"].unique())
    left_out = c2.selectbox("Robustness check: leave out one lab batch", ["None"] + runs,
                            help="Re-runs the test without that batch. If a result stays coloured, "
                                 "it isn't caused by one batch's quirks.")
    drop = () if left_out == "None" else (left_out,)
    same_batch = c2.checkbox("Only people whose samples were all processed in one batch",
                             help="For these people, any batch quirk hits their flare and calm samples equally, "
                                  "so it cancels out in the comparison.")
    steroid_win = "None"
    if "steroid_hours" in D["samples"]:
        steroid_win = c2.selectbox("Robustness check: leave out samples taken soon after (recorded) steroids",
                                   list(STEROID_WINDOWS),
                                   help="Steroids treat flares and change immune cells. If a result holds without "
                                        "samples with recorded recent steroids, recorded steroid use isn't what's "
                                        "driving it. " + STEROID_CAVEAT)
    res, pairs = paired_cell_stats(level, group, drop, same_batch, None, STEROID_WINDOWS[steroid_win])
    if steroid_win != "None":
        what = "with **any recorded steroid use**" if steroid_win.startswith("Any") else \
            f"taken within **{steroid_win}** of recorded steroids"
        st.caption(f"Leaving out samples {what}. Steroid records may be incomplete (AI-extracted, not checked).")
    if drop:
        st.caption(f"Showing results **without batch {left_out}**.")
    if res.empty:
        st.warning("Not enough people with both kinds of sample for this selection.")
        st.stop()
    event = chart(signature_bar(res), on_select="rerun", selection_mode="points",
                  key=f"sig_bar_{level}_{group}_{left_out}_{same_batch}_{steroid_win}")
    st.caption("Coloured bars pass a strict test (adjusted p < 0.05, corrected for testing many cell types). "
               "Labels show how many people had more of that cell type during their flare. "
               "👆 Click a bar to see it person by person.")
    types = res["cell type"].tolist()
    clicked = next((p.get("y") for p in (event.selection.points if event else []) if p.get("y") in types), None)
    if clicked and clicked != st.session_state.get("sig_last"):
        st.session_state["sig_ct"] = st.session_state["sig_last"] = clicked
    if st.session_state.get("sig_ct") not in types:
        st.session_state.pop("sig_ct", None)

    st.subheader("See it person by person")
    ct = st.selectbox("Cell type", types, key="sig_ct")
    row = res[res["cell type"] == ct].iloc[0]
    direction, consistent = verdict(row)
    st.markdown(f"**{ct}** {direction} during a flare in **{consistent} of {row['people']}** people "
                f"(adjusted p {pval_text(row['adjusted p'])}).")
    chart(slope_chart(pairs[pairs["cell_type"] == ct], "% of immune cells", ct))
    with st.expander("Full table"):
        st.dataframe(res.round(3), hide_index=True, width="stretch")

elif page == "Seeing a flare coming":
    from scipy.stats import mannwhitneyu, spearmanr

    header("Can the blood show a flare coming?")
    st.markdown("We score each blood sample for how flare-like it looks. Then we ask: "
                "**do calm samples taken shortly before a flare already look flare-like?**")
    SCORES = {
        "Personal B-cell change (vs your own calm level)":
            ("bcell", "Change in B cells vs own calm level (percentage points)",
             "Each sample's B-cell share minus that person's average in their *other* calm samples. No model, "
             "nothing to overfit."),
        "Personal flare score (all cell types, vs your own calm level)":
            ("model", "Personal flare score (0 = calm-like, 1 = flare-like)",
             "Every cell type's change from the person's own calm level, combined by a model that never saw "
             "that person."),
        "General flare score (same rule for everyone)":
            ("general", "General flare score (0 = calm-like, 1 = flare-like)",
             "One rule for everyone, no personal baseline. Each person is scored by a model that never saw them."),
    }
    choice = st.radio("Score", list(SCORES), horizontal=False)
    kind, score_label, explain = SCORES[choice]
    st.caption(explain)
    if kind == "general":
        scored, weights, auc = flare_scores(group)
    else:
        scored, weights, auc = personal_scores(group, kind)

    with st.expander("Compare all three scores"):
        rows = []
        for name, (k, _, _) in SCORES.items():
            a_ = flare_scores(group)[2] if k == "general" else personal_scores(group, k)[2]
            rows.append({"score": name, "correct ranking": f"{a_:.0%}" if a_ == a_ else "n/a"})
        st.table(pd.DataFrame(rows).set_index("score"))

    st.subheader("1. Can the score tell flare blood from calm blood?")
    c1, c2 = st.columns([1, 2])
    c1.metric("Correct ranking", f"{auc:.0%}" if auc == auc else "n/a",
              help="Pick one flare sample and one calm sample at random: how often does the flare sample get "
                   "the higher score? 50% = coin toss, 100% = perfect.")
    c1.caption("50% would be a coin toss, 100% perfect.")
    fig = px.strip(scored.dropna(subset=["score"]), x="state", y="score", color="state",
                   color_discrete_map=STATE_COLORS, hover_data={"person": True, "day": ":.0f"},
                   labels={"state": "", "score": score_label})
    fig.update_layout(title="Flare score by sample status", height=320, showlegend=False,
                      xaxis_title="Sample status")
    chart(fig, where=c2)
    if weights is not None:
        with st.expander("Which cell types drive the score?"):
            w = pd.concat([weights.head(5), weights.tail(5)]).reset_index()
            w.columns = ["cell type", "weight"]
            w["pushes toward"] = np.where(w["weight"] > 0, "Flare", "Feeling fine")
            fig = px.bar(w, x="weight", y="cell type", orientation="h", color="pushes toward",
                         color_discrete_map={"Flare": FLARE, "Feeling fine": FINE}, labels={"cell type": ""})
            fig.update_layout(title="Cell types that drive the flare score", height=360, legend_title_text="")
            chart(fig)

    st.subheader("2. Countdown: calm samples before a flare")
    cd = countdown(scored)
    window = st.slider("Count a flare as 'coming' if it happened within this many days", 14, 120, 45, step=7)
    coming = cd.dropna(subset=["days_to_flare", "score"])
    if len(coming) >= 4:
        rho, p_rho = spearmanr(coming["days_to_flare"], coming["score"])
        fig = px.scatter(coming, x="days_to_flare", y="score", color="person",
                         hover_data={"day": ":.0f", "days_to_flare": ":.0f"},
                         labels={"days_to_flare": "Days until this person's next flare",
                                 "score": score_label + " (calm sample)", "person": ""})
        fit_ = np.polyfit(coming["days_to_flare"], coming["score"], 1)
        xs = np.linspace(coming["days_to_flare"].min(), coming["days_to_flare"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(fit_, xs), mode="lines", name="Trend",
                                 line=dict(color=avg_color(), width=3, dash="dash")))
        fig.add_vrect(x0=0, x1=window, fillcolor=FLARE, opacity=0.08, line_width=0, layer="below",
                      annotation_text="flare coming", annotation_position="top left")
        fig.update_xaxes(autorange="reversed")
        fig.update_layout(title="Calm sample scores as the next flare approaches", height=430)
        chart(fig)
        trend = "rises" if rho < 0 else "falls"
        st.markdown(f"Read right to left: as a flare gets closer, the score **{trend}** "
                    f"(correlation {-rho:+.2f}, p {pval_text(p_rho)}, {len(coming)} calm samples "
                    f"from {coming['person'].nunique()} people).")
    else:
        st.info("Too few calm samples followed by a flare in this group to draw the countdown.")

    st.subheader(f"3. Calm samples with a flare within {window} days vs the rest")
    cd["kind"] = np.select(
        [cd["days_to_flare"] <= window, ~cd["is_last"]],
        [f"Flare within {window} days", "No flare that soon"], default="Unknown (last sample)")
    comp = cd[(cd["kind"] != "Unknown (last sample)") & cd["score"].notna()]
    a = comp.loc[comp["kind"] == f"Flare within {window} days", "score"]
    b = comp.loc[comp["kind"] == "No flare that soon", "score"]
    if len(a) >= 3 and len(b) >= 3:
        p_mw = mannwhitneyu(a, b, alternative="greater").pvalue
        fig = px.box(comp, x="kind", y="score", points="all", color="kind",
                     color_discrete_map={f"Flare within {window} days": FLARE, "No flare that soon": FINE},
                     hover_data={"person": True, "days_to_flare": ":.0f"},
                     labels={"kind": "", "score": score_label + " (calm sample)"})
        fig.update_layout(title=f"Scores in calm samples: flare within {window} days or not",
                          height=380, showlegend=False, xaxis_title="Next flare timing")
        chart(fig)
        st.markdown(f"Calm samples with a flare coming: median score **{a.median():.2f}** ({len(a)} samples) vs "
                    f"**{b.median():.2f}** for the rest ({len(b)} samples); "
                    f"one-sided test p {pval_text(p_mw)}.")
    else:
        st.info("Too few calm samples in one of the two groups for this window. Try a different window.")

    if kind == "bcell":
        st.subheader("4. A red flag: a big jump above your own normal")
        st.markdown("Most flares barely move the B-cell share. But does a **big** jump above a person's own calm "
                    "level ever happen when they feel fine?")
        flagged_data = scored.dropna(subset=["score"])
        limit = st.slider("Red flag when B cells are more than this many points above the person's own normal",
                          5, 30, 15)
        fl = flagged_data[flagged_data["flare"] == "Flaring"]
        ca = flagged_data[flagged_data["flare"] == "Baseline"]
        fl_hit, ca_hit = fl[fl["score"] > limit], ca[ca["score"] > limit]
        c = st.columns(3)
        c[0].metric("Flare samples flagged", f"{len(fl_hit)} of {len(fl)}")
        c[1].metric("Calm samples flagged (false alarms)", f"{len(ca_hit)} of {len(ca)}")
        c[2].metric("Different people among flagged flares", fl_hit["person"].nunique())
        fig = px.strip(flagged_data, x="state", y="score", color="state", color_discrete_map=STATE_COLORS,
                       hover_data={"person": True, "day": ":.0f"},
                       labels={"state": "", "score": "Change in B cells vs own calm level (percentage points)"})
        fig.add_hline(y=limit, line_dash="dash", line_color=avg_color(),
                      annotation_text=f"red flag: +{limit} points", annotation_position="top right")
        fig.update_layout(title="B cell changes in flare and calm samples", height=380, showlegend=False,
                          xaxis_title="Sample status")
        chart(fig)
        if len(fl_hit):
            alarm = "no false alarms" if len(ca_hit) == 0 else f"{len(ca_hit)} false alarm(s)"
            st.markdown(f"At +{limit} points the red flag catches **{len(fl_hit)} of {len(fl)} flares** "
                        f"(from {fl_hit['person'].nunique()} people) with **{alarm}** among {len(ca)} calm samples. "
                        "It misses most flares, but when it fires, it's a flare.")
        st.caption("The line was picked after looking at this data, so it needs confirming on new patients.")

    st.subheader("5. A trained flare classifier (tested on people it never saw)")
    clf_file = f"{DATA}/classifier/results.json"
    if not os.path.exists(clf_file):
        st.info("Not built yet. In SageMaker run `python flare_classifier.py` (about 10–25 minutes), then refresh.")
    else:
        with open(clf_file) as f:
            clf = json.load(f)
        st.markdown(f"Each blood sample is a bag of thousands of cells with one label (flare or calm). Simple models "
                    f"learned from **{clf['samples']} samples of {clf['people']} {clf['group'].lower()} patients**, "
                    "and each person was scored by a model that **never saw them**. The **shuffle test** scrambles "
                    f"the flare labels within each person {clf['shuffles']} times and retrains: a real result should "
                    "beat the shuffled ones.")
        rows = []
        for name, r in clf["results"].items():
            passed = r["shuffle_p"] < 0.05
            rows.append({"approach": name, "correct ranking": r["correct_ranking"],
                         "flare scored higher": f"{r['people_flare_higher']} of {r['people_with_both']} people",
                         "shuffled labels (average)": r["shuffle_mean"], "shuffle-test p": r["shuffle_p"],
                         "beats chance?": "✅ yes" if passed else "❌ no"})
        tab = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tab["approach"], y=tab["correct ranking"], name="Real labels",
                             marker_color=[FLARE if b.startswith("✅") else GREY for b in tab["beats chance?"]],
                             text=[f"{v:.0%}" for v in tab["correct ranking"]], textposition="outside"))
        fig.add_trace(go.Scatter(x=tab["approach"], y=[r["shuffle_95th"] for r in clf["results"].values()],
                                 mode="markers", name="Shuffled labels: 95% of runs below this",
                                 marker=dict(symbol="line-ew-open", size=40, line=dict(width=3, color=avg_color()))))
        fig.add_hline(y=0.5, line_dash="dot", line_color=GREY, annotation_text="coin toss", annotation_position="left")
        fig.update_layout(title="How often each method ranks flare samples higher",
                          height=380, xaxis_title="Scoring method",
                          yaxis=dict(range=[0, 1.08], tickformat=".0%", title="Correct ranking"),
                          legend=dict(orientation="h", y=1.12))
        chart(fig)
        st.dataframe(tab.style.format({"correct ranking": "{:.0%}", "shuffled labels (average)": "{:.0%}",
                                       "shuffle-test p": "{:.3f}"}), hide_index=True, width="stretch")
        best = max(clf["results"].items(), key=lambda kv: kv[1]["correct_ranking"])
        verdict = "beats" if best[1]["shuffle_p"] < 0.05 else "does not clearly beat"
        st.markdown(f"Best: **{best[0]}** at **{best[1]['correct_ranking']:.0%}**, which {verdict} shuffled labels "
                    f"(p = {best[1]['shuffle_p']:.3f}).")
        with st.expander("What each model leans on"):
            for name, r in clf["results"].items():
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame(r["top_features"]).round(3), hide_index=True, width="stretch")
        st.caption(f"Gene-activity models use the {clf['top_genes_per_round']} most flare-related genes, picked again in "
                   f"every round from the training people only, across these cell types: "
                   f"{', '.join(clf['cell_types_for_genes'])}. With ~30 people, treat results as a first look; ImYoo's "
                   "held-out patients would be the real test.")

    st.subheader("⚠️ Read this as a first look")
    st.markdown("- Only a few calm samples were taken shortly before a flare, so this is **exploratory**.\n"
                "- Samples from the same person aren't fully independent; p-values here are optimistic.\n"
                "- A real early-warning test would need more people and confirmation in ImYoo's held-out patients.")

elif page == "Ask the Flare Analyst":
    import analyst

    header("Ask the Flare Analyst")
    st.markdown("Ask a question in plain English. **Claude** (running inside AWS) picks the right analysis, runs it on "
                "the study's summary numbers, shows the chart and explains what it means, honestly.")
    c1, c2, c3 = st.columns([2, 1, 1])
    model_name = c1.selectbox("Model", list(analyst.MODELS))
    if c3.button("Clear conversation"):
        st.session_state.pop("chat", None)
        st.session_state.pop("bedrock_messages", None)
    st.caption("Claude only sees summary results (counts, averages, test results), never cells or raw data. "
               "People appear only as P01, P02, …")
    chat = st.session_state.setdefault("chat", [])
    history = st.session_state.setdefault("bedrock_messages", [])

    def show(item):
        if item["kind"] == "user":
            st.chat_message("user").markdown(item["text"])
        elif item["kind"] == "text":
            st.chat_message("assistant", avatar="🩸").markdown(item["text"])
        else:
            with st.chat_message("assistant", avatar="🔬"):
                st.markdown(f"_{analyst.describe_call(item['name'], item['input'])}_")
                if item["fig"] is not None:
                    chart(item["fig"], key=f"fig{item['n']}")
                with st.expander("Numbers Claude saw"):
                    st.json(item["result"])

    for item in chat:
        show(item)

    examples = ["Do B cells rise in Crohn's patients and in ulcerative colitis patients separately?",
                "Which genes switch on in B cells during a flare?",
                "Is the B-cell finding a batch effect?",
                "Could a personal blood score warn of a flare before it happens?"]
    st.markdown("**Try:** " + " · ".join(f"*{e}*" for e in examples))
    auto = c2.button("🔁 Explore on its own", help="Claude runs hypothesis → test → result → next question by itself.")
    question = st.chat_input("Ask about flares, cell types, genes, batches…")
    if auto:
        question = analyst.AUTOPILOT.format(rounds=3)
    if question:
        item = {"kind": "user", "text": "🔁 *Explore on its own (3 rounds)*" if auto else question}
        chat.append(item)
        show(item)
        history.append({"role": "user", "content": [{"text": question}]})

        def on_event(kind, payload):
            item = ({"kind": "text", "text": payload} if kind == "text"
                    else {"kind": "tool", "n": len(chat), **payload})
            chat.append(item)
            show(item)

        try:
            with st.spinner("Claude is analysing…"):
                analyst.ask(analyst.client(), analyst.MODELS[model_name], history, analyst_tools(), on_event,
                            max_steps=20 if auto else 10)
        except Exception as e:
            history.pop()
            st.error(f"Couldn't reach Claude: {type(e).__name__}: {e}")

elif page == "My immune report":
    header("My immune report")
    st.caption("A patient-friendly view: each blood sample compared with **your own normal**, not with other people. "
               "Research prototype from a small study. Not medical advice.")
    X, delta, normal, calm_d = personal_family_deltas()
    people = sorted(samples_g["person"].unique())
    multi = samples_g.groupby("person")["flare"].nunique().eq(2)
    default = next((p for p in people if multi.get(p, False)), people[0])
    c1, c2 = st.columns(2)
    if st.session_state.get("report_person") not in people:
        st.session_state.pop("report_person", None)
    person = c1.selectbox("Person", people, key="report_person",
                          **({} if "report_person" in st.session_state else {"index": people.index(default)}))
    s = D["samples"][D["samples"]["person"] == person].sort_values("day")
    labels = sample_labels(s)
    by_label = {v: k for k, v in labels.items()}
    if st.session_state.get("report_sample") not in by_label:
        st.session_state.pop("report_sample", None)
    sample = by_label[c2.selectbox("Blood sample", list(by_label), key="report_sample",
                                   **({} if "report_sample" in st.session_state else {"index": len(by_label) - 1}))]
    row = s[s["sample_id"] == sample].iloc[0]
    d = delta.loc[sample]
    if d.isna().all():
        st.info(f"{person} needs at least one *other* feeling-fine sample to set their personal normal, "
                "so this sample can't be compared yet.")
        st.stop()

    LO, HI = 0.05, 0.95  # "typical calm day" = middle 90% of calm samples' changes from their own normal

    def rank(f, v):
        ref = calm_d[f].dropna()
        return 100 * (ref < v).mean() if len(ref) else 50.0

    def band(f):
        ref = calm_d[f].dropna()
        return (ref.quantile(LO), ref.quantile(HI)) if len(ref) else (0.0, 0.0)

    MAJOR = ["T Cells", "NK Cells", "B Cells", "Monocytes"]
    fams = [f for f in d.index if d[f] == d[f]]
    outside = [f for f in fams if not band(f)[0] <= d[f] <= band(f)[1]]
    b = d.get("B Cells", np.nan)
    b_max = calm_d["B Cells"].max() if "B Cells" in calm_d else np.nan
    if b == b and b > b_max:
        st.error(f"### 🚩 B cells are {b:+.0f} points above {person}'s normal\n"
                 f"That's higher than **every** feeling-fine sample in the study (the highest was {b_max:+.0f}). "
                 "Jumps like this were seen only during flares.")
    elif b == b and rank("B Cells", b) >= 100 * HI:
        st.warning(f"### ⚠️ B cells unusually high for a calm day\n{b:+.0f} points above {person}'s normal: "
                   f"higher than {rank('B Cells', b):.0f}% of feeling-fine samples in the study.")
    elif [f for f in outside if f in MAJOR]:
        shifted = ", ".join(f"{f} {'up' if d[f] > 0 else 'down'}" for f in outside if f in MAJOR)
        st.warning(f"### ⚠️ Something has shifted\nUnusual for a calm day: **{shifted}**.")
    else:
        st.success(f"### ✅ Like a typical calm day for {person}\nEvery main cell family is within the range seen on "
                   "feeling-fine days across the study.")
    st.caption(f"Condition: {row['disease']} · this sample: {labels[sample]} · "
               f"{int((s['flare'] == 'Baseline').sum())} feeling-fine samples set the personal normal.")

    st.subheader("This sample compared with your normal")
    fams = sorted(fams, key=lambda f: -normal.loc[person, f] if person in normal.index else 0)
    fig = go.Figure()
    for i, f in enumerate(fams):
        lo, hi = band(f)
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=i - 0.35, y1=i + 0.35, fillcolor=GREY, opacity=0.35,
                      line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=[d[f] for f in fams], y=list(range(len(fams))), mode="markers",
        marker=dict(size=16, color=[FLARE if f in outside else FAMILY_COLORS.get(f, "#888") for f in fams],
                    line=dict(width=1, color="white")),
        customdata=[[f, rank(f, d[f])] for f in fams], showlegend=False,
        hovertemplate="%{customdata[0]}: %{x:+.1f} points vs normal<br>"
                      "higher than %{customdata[1]:.0f}% of calm samples<extra></extra>"))
    fig.add_vline(x=0, line_color=avg_color(), line_width=1)
    fig.update_yaxes(tickvals=list(range(len(fams))), ticktext=fams, autorange="reversed")
    fig.update_layout(title="This sample versus your usual cell mix", height=80 + 48 * len(fams),
                      xaxis_title="Change from your normal (percentage points)")
    chart(fig)
    st.caption("Grey band = a typical calm day: where 90% of feeling-fine samples in the study fall, each compared "
               "with that person's own normal. Orange dots are outside it. Hover a dot for its percentile.")

    st.subheader(f"{person}'s B cells over time")
    if "B Cells" in X.columns and person in normal.index:
        ts = s[["sample_id", "day", "state"]].copy()
        ts["b"] = X.loc[ts["sample_id"], "B Cells"].to_numpy()
        base = normal.loc[person, "B Cells"]
        lo, hi = band("B Cells")
        fig = go.Figure()
        fig.add_hrect(y0=base + lo, y1=base + hi, fillcolor=GREY, opacity=0.3, line_width=0, layer="below",
                      annotation_text="typical calm day", annotation_position="top left")
        fig.add_hline(y=base + b_max, line_dash="dash", line_color=FLARE,
                      annotation_text="red flag: above every calm sample", annotation_position="top left")
        fig.add_trace(go.Scatter(x=ts["day"], y=ts["b"], mode="lines", line=dict(color=GREY, width=2),
                                 showlegend=False, hoverinfo="skip"))
        for state, grp in ts.groupby("state"):
            fig.add_trace(go.Scatter(x=grp["day"], y=grp["b"], mode="markers", name=state,
                                     marker=dict(size=13, color=STATE_COLORS[state]),
                                     hovertemplate="Day %{x:.0f}: %{y:.1f}% B cells<extra></extra>"))
        sel = ts[ts["sample_id"] == sample]
        fig.add_trace(go.Scatter(x=sel["day"], y=sel["b"], mode="markers", name="this sample",
                                 marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color=avg_color()))))
        fig.update_layout(title=f"{person}'s B cell share across blood samples", height=380,
                          xaxis_title="Days since first sample", yaxis_title="B cells (% of immune cells)",
                          legend=dict(orientation="h", y=1.12))
        chart(fig)

    facts = [f"Person: {person}; condition: {row['disease']}",
             f"This sample: {labels[sample]}",
             f"Compared with their own normal from {int((s['flare'] == 'Baseline').sum())} feeling-fine samples"]
    facts += [f"{f}: {d[f]:+.1f} percentage points vs their normal, higher than {rank(f, d[f]):.0f}% of "
              f"feeling-fine samples in the study ({'unusual' if f in outside else 'typical'} for a calm day)"
              for f in fams]
    all_fl = delta.loc[D["samples"].set_index("sample_id")["flare"].eq("Flaring").reindex(delta.index).fillna(False)]
    n_hit = int((all_fl["B Cells"] > b_max).sum()) if "B Cells" in all_fl else 0
    facts.append(f"In this study, B-cell jumps above every feeling-fine sample (more than {b_max:+.0f} points) were "
                 f"seen in {n_hit} of {int(all_fl['B Cells'].notna().sum())} flare samples.")
    if st.button("💬 Explain my report in plain words"):
        import analyst
        try:
            with st.spinner("Claude is writing a plain-language summary…"):
                text = analyst.explain_report(analyst.client(), analyst.MODELS["Claude Sonnet 4.6 (fast)"], facts)
            st.info(text)
        except Exception as e:
            st.error(f"Couldn't reach Claude: {type(e).__name__}: {e}")
    with st.expander("The numbers behind this report"):
        st.markdown("\n".join(f"- {f}" for f in facts))

elif page == "Sid's journey":
    header("Sid's journey: one patient's T cells through cancer treatment",
           "Sid Sijbrandij openly shared his osteosarcoma data to advance cancer research. Here we follow the "
           "**T cells in his blood** across a year of treatment, using the tools we built for ImYoo.")
    SD = load_sid()
    if SD is None:
        st.info("Sid's summaries aren't built yet. In SageMaker run `python sid_precompute.py` (see README), "
                "then refresh this page.")
        st.stop()
    si, events, comp = SD["info"], SD["events"], SD["composition"]
    dv, cl, draws = SD["tcell_diversity"], SD["clones"], SD["draws"].dropna(subset=["date"])
    start, end = comp["date"].min(), comp["date"].max()
    span = (pd.Timestamp(si["last_draw"]) - pd.Timestamp(si["first_draw"])).days
    broad = comp[comp["level"] == "broad"]
    t_share = broad[broad["cell_type"] == "T Cells"]["pct"].mean() if (broad["cell_type"] == "T Cells").any() else 0
    kpis([("🩸", si["draws"], "blood draws"), ("📅", f"{span} days", f"{si['first_draw']} → {si['last_draw']}"),
          ("🔬", f"{si['cells']:,}", "immune cells"),
          ("🧬", f"{int(dv['t_cells_with_receptor'].sum()):,}", "T cells with a receptor barcode")])
    if t_share > 85:
        finding("What this blood collection is",
                f"Almost all of Sid's cells are **T cells (about {t_share:.0f}%)**, compared with about half in "
                "typical blood",
                "So this collection appears to be focused on T cells. That makes it ideal for following T-cell "
                "families over time, but the overall cell mix mostly reflects how samples were collected, not biology.",
                quiet=True)

    st.subheader("T-cell families over time")
    st.markdown("Each T cell carries a receptor 'barcode', and cells with the same barcode belong to one **family**. "
                "When the immune system responds to something (an infection, a tumour, a treatment), particular "
                "families **multiply**. Dashed lines mark tumour samples taken in the same period.")
    c1, c2 = st.columns([2, 1])
    top_n = c2.slider("Families to show", 3, 15, 8)
    order = cl.groupby("family")["pct"].max().sort_values(ascending=False).index[:top_n]
    fig = px.line(cl[cl["family"].isin(order)].sort_values("date"), x="date", y="pct", color="family", markers=True,
                  hover_data={"clone": True}, labels={"date": "Blood draw date", "pct": "Share of T cells (%)",
                                                       "family": "T cell family"})
    mark_events(fig, events, start, end)
    fig.update_layout(title="T cell families across blood draws", height=420)
    chart(fig, where=c1)
    fig = px.line(dv.sort_values("date"), x="date", y="effective_families", markers=True,
                  labels={"date": "Blood draw date", "effective_families": "Effective number of T cell families"})
    fig.update_traces(line_color=FINE)
    mark_events(fig, events, start, end)
    fig.update_layout(title="T cell family diversity over time", height=280)
    chart(fig, where=c2)
    c2.caption("Falls when a few families take over (a focused response), rises when many share the space.")

    st.subheader("Which families grew or shrank?")
    in_range = events[(events["date"] > start) & (events["date"] < end)]
    split_opts = {f"{r['label']} ({r['date']:%Y-%m-%d})": r["date"] for _, r in in_range.iterrows()}
    split_opts["Halfway through the draws"] = start + (end - start) / 2
    c1, c2 = st.columns([2, 1])
    split_name = c1.selectbox("Compare blood draws before and after", list(split_opts))
    split = split_opts[split_name]
    before, after = cl[cl["date"] < split], cl[cl["date"] > split]
    if before["date"].nunique() and after["date"].nunique():
        ba = pd.DataFrame({"before": before.groupby("family")["pct"].mean(),
                           "after": after.groupby("family")["pct"].mean()}).fillna(0)
        ba["change"] = ba["after"] - ba["before"]
        ba = ba.loc[ba["change"].abs().sort_values(ascending=False).index[:12]].sort_values("change")
        ba["direction"] = np.where(ba["change"] > 0, "Grew after", "Shrank after")
        fig = px.bar(ba.reset_index(), x="change", y="family", orientation="h", color="direction",
                     color_discrete_map={"Grew after": FLARE, "Shrank after": FINE},
                     hover_data={"before": ":.2f", "after": ":.2f"},
                     labels={"change": "Change in T cell share (percentage points)", "family": "T cell family"})
        fig.update_layout(title="Change in T cell family share after the selected date",
                          height=420, legend_title_text="")
        chart(fig, where=c1)
        nb, na = before["date"].nunique(), after["date"].nunique()
        c2.metric("Draws before", nb)
        c2.metric("Draws after", na)
        grew = ba[ba["change"] > 0]
        if len(grew):
            g = grew["change"].idxmax()
            c2.markdown(f"Biggest grower: **{g}**, from {ba.loc[g, 'before']:.2f}% to {ba.loc[g, 'after']:.2f}% of his "
                        "T cells.")
        c2.caption("Averages of the draws on each side. One person, so this describes his response; it can't say "
                   "what caused it.")
    else:
        st.info("Need blood draws on both sides of that date.")

    st.subheader("Blood draws")
    fig = px.bar(draws.sort_values("date"), x="date", y="n_cells",
                 labels={"date": "Blood draw date", "n_cells": "Cells captured"})
    fig.update_traces(marker_color=GREY)
    mark_events(fig, events, start, end)
    fig.update_layout(title="Cells captured at each blood draw", height=260)
    chart(fig)
    notes = []
    if si.get("dates_unreadable"):
        notes.append(f"{si['dates_unreadable']:,} cells have no date in the file (they come from a separately named "
                     "sample) and are left out of the timeline.")
    groups = SD.get("groups")
    if groups is not None and (groups["group"] == "").any():
        n_nog = int(groups.loc[groups["group"] == "", "cells"].sum())
        notes.append(f"{n_nog:,} cells have no group in the file; they were labelled together as one group.")
    if notes:
        st.caption(" ".join(notes))

    with st.expander("Overall cell mix at each draw (read with care)"):
        st.caption("Because the collection is focused on T cells, these shares mostly reflect how samples were "
                   "collected, not changes in his blood.")
        level = st.radio("Detail", ["broad", "detailed"], horizontal=True, key="sid_level",
                         format_func={"broad": "Broad cell families", "detailed": "Detailed cell types"}.get)
        c = comp[comp["level"] == level]
        fig = px.area(c.sort_values("date"), x="date", y="pct", color="cell_type", markers=True,
                      color_discrete_map=type_colors(level), color_discrete_sequence=DETAIL_PALETTE,
                      labels={"date": "Blood draw date", "pct": "Immune cells (%)", "cell_type": "Cell type"})
        mark_events(fig, events, start, end)
        fig.update_layout(title="Immune cell composition across blood draws", height=420,
                          legend=dict(orientation="h", y=-0.15))
        chart(fig)

    st.subheader("His cell map")
    cm = SD["cells_map"].dropna(subset=["date"]).copy()
    cm["draw"] = cm["date"].dt.strftime("%Y-%m-%d")
    colour = st.radio("Colour by", ["Blood draw", "Cell type", "Detailed cell type"], horizontal=True, key="sid_map")
    col = {"Cell type": "broad", "Detailed cell type": "detailed", "Blood draw": "draw"}[colour]
    fig = px.scatter(cm, x="x", y="y", color=col, render_mode="webgl",
                     color_discrete_map=FAMILY_COLORS if col == "broad" else None,
                     color_discrete_sequence=DETAIL_PALETTE, labels={col: ""},
                     category_orders={"draw": sorted(cm["draw"].unique())})
    fig.update_traces(marker=dict(size=2.5, opacity=0.6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Map of cells, grouped by similarity", height=600,
                      legend=dict(itemsizing="constant"))
    chart(fig)

    st.subheader("How the cells were labelled (honestly)")
    st.markdown("Sid's file has no cell-type labels. We first tried teaching a model on **ImYoo's expert-labelled "
                "cells** and applying it to Sid's. But the two labs used **different kits**, and on its own the model "
                "wasn't reliable. So each **group of similar cells** is labelled with three clues: **(1)** the share "
                "carrying a **T-cell receptor** (only T cells have one; this did most of the work), **(2)** the "
                "model's **average vote**, **(3)** **marker genes**.")
    c = st.columns(3)
    if si.get("model_alone_tcr_check_pct") is not None:
        c[0].metric("ImYoo model alone: receptor cells called T", f"{si['model_alone_tcr_check_pct']:.0f}%",
                    help="Too low to trust, which is why we label groups using the T-cell receptors.")
    if si.get("tcr_in_t_groups_pct") is not None:
        c[1].metric("Receptor-carrying cells in T-cell groups", f"{si['tcr_in_t_groups_pct']:.0f}%")
    if si.get("model_group_agreement_pct") is not None:
        c[2].metric("Model agrees with group label", f"{si['model_group_agreement_pct']:.0f}%")
    st.caption("Lesson for future work: carrying labels across different lab kits needs stronger methods that line "
               "the datasets up first. T-cell receptor barcodes were the reliable anchor here.")
    if groups is not None:
        with st.expander("How each group was labelled"):
            gt = groups.assign(group=groups["group"].replace("", "(no group in file)")).rename(columns={
                "t_receptor_pct": "% with T-cell receptor", "model_top": "model's top vote", "model_vote_pct": "vote %",
                "reason": "why", "top_markers": "strongest markers"})
            st.dataframe(gt.round(1), hide_index=True, width="stretch")
    mk = SD["markers"].set_index("predicted family")
    fig = px.imshow(mk, aspect="auto", color_continuous_scale="Oranges",
                    labels=dict(x="Marker gene", y="Labelled as", color="Activity"))
    fig.update_layout(title="Marker activity supporting each cell group label", height=280)
    chart(fig)

    st.subheader("⚠️ Keep in mind")
    st.markdown("- **One person:** this describes Sid's journey; it isn't a finding about osteosarcoma in general.\n"
                "- **A T-cell-focused collection:** the overall cell mix can't be read as changes in his blood.\n"
                "- **Treatment dates aren't in the data**, so changes can't be tied to specific treatments, only "
                "placed relative to his tumour sampling dates.")

elif page == "Compare conditions":
    header("IBD vs cancer: do their immune changes look alike?",
           "We compare **how genes change** rather than raw gene levels. A change is measured within the same person "
           "(and the same lab kit), so differences between people and kits largely cancel out.")
    st.caption("Changes are in log2 units: +1 means about twice as active, −1 about half. Exploratory: few people "
               "per group, so read patterns, not single genes.")
    pb_types = sorted(D["pseudobulk"]["cell_type"].unique())

    st.subheader("1. Bowel-disease flares vs Sid's cancer journey")
    st.markdown("Do immune cells change in a similar way during an **IBD flare** (Crohn's, ulcerative colitis) and across **Sid's osteosarcoma treatment**? For Sid, the change is his blood draws after vs before a chosen date.")
    SD = load_sid()
    if SD is None or SD.get("pseudobulk") is None:
        st.info("Sid's gene summaries aren't built yet. In SageMaker run `python sid_pseudobulk.py` (after "
                "`sid_precompute.py`), then refresh.")
        SD = None
    def sid_vs_ibd():
        sp, events = SD["pseudobulk"], SD["events"]
        sid_types = sorted(sp["cell_type"].unique())
        c1, c2, c3 = st.columns(3)
        pick = next((t for t in ["CD8 T Cells", "T Cells"] if t in sid_types), sid_types[0])
        sid_ct = c1.selectbox("Sid's cells", sid_types, index=sid_types.index(pick))
        t_like = [c for c in pb_types if family_of(c) == family_of(sid_ct)] or pb_types
        same = next((c for c in t_like if c.lower() == sid_ct.lower()),
                    next((c for c in t_like if "cd8" in c.lower() and "cd8" in sid_ct.lower()), t_like[0]))
        ibd_ct = c2.selectbox("Compare with IBD cell type", t_like, index=t_like.index(same), key="cmp_ibd_ct")
        start, end = sp["date"].min(), sp["date"].max()
        in_range = events[(events["date"] > start) & (events["date"] < end)]
        split_opts = {f"{r['label']} ({r['date']:%Y-%m-%d})": r["date"] for _, r in in_range.iterrows()}
        split_opts["Halfway through his draws"] = start + (end - start) / 2
        surgery = next((i for i, k in enumerate(split_opts) if "resection" in k.lower() or "surgery" in k.lower()),
                       len(split_opts) - 1)
        split_name = c3.selectbox("Sid: compare draws after vs before", list(split_opts), index=surgery)
        split = split_opts[split_name]
        one = sp[sp["cell_type"] == sid_ct]
        genes_sid = [g for g in one.columns if g not in ("date", "cell_type", "n_cells")]
        before, after = one[one["date"] < split], one[one["date"] > split]
        ibd = flare_gene_changes(ibd_ct)
        if before.empty or after.empty or ibd.empty:
            st.info("Need Sid's blood draws on both sides of that date, and IBD patients with both kinds of sample.")
            return
        sid_change = after[genes_sid].mean() - before[genes_sid].mean()
        # match gene names: ImYoo may use gene names or gene codes
        names = SD.get("gene_names")
        overlap_names = len(set(sid_change.index) & set(ibd.index))
        if names is not None and "code" in names and overlap_names < 500:
            code_of = dict(zip(names["name"], names["code"]))
            sid_change.index = [code_of.get(g, g) for g in sid_change.index]
            sid_change = sid_change[~sid_change.index.duplicated()]
        shared = ibd.index.intersection(sid_change.index)
        if len(shared) < 20:
            st.warning(f"Only {len(shared)} genes could be matched between the two studies.")
            return
        ibd_s = ibd.loc[shared]
        keep = ibd_s["activity"] >= ibd_s["activity"].quantile(0.25)
        df = pd.DataFrame({"x": ibd_s.loc[keep, "change"], "y": sid_change.loc[shared][keep.values]})
        df.index.name = "gene"
        fig, rho, df = compare_scatter(df, "IBD flare", "Sid", f"Change during IBD flares ({ibd_ct})",
                                       f"Sid's change after vs before ({sid_ct})")
        c = st.columns(3)
        c[0].metric("Similarity", f"{rho:+.2f}",
                    help=f"Spearman correlation of changes in the {df.attrs['n_moving']} genes that clearly change in at "
                         f"least one condition (+1 same pattern, 0 unrelated, −1 opposite). All genes: "
                         f"{df.attrs['rho_all']:+.2f}.")
        c[1].metric("Sid's draws", f"{before['date'].nunique()} before · {after['date'].nunique()} after")
        c[2].metric("Genes compared", f"{len(df):,}")
        st.markdown(f"Sid's **{sid_ct}** change around *{split_name}* and the change in **{ibd_ct}** during IBD flares "
                    f"are **{similarity_words(rho)}** (similarity {rho:+.2f}, based on {df.attrs['basis']}).")
        chart(fig)
        cols = st.columns(2)
        for col, (title, pats) in zip(cols, [("Moving the same way", ["Up in both", "Down in both"]),
                                             ("Moving in opposite directions", ["Opposite directions"])]):
            sel = df[df["pattern"].isin(pats)]
            top = sel.assign(order=(sel["x"].abs() + sel["y"].abs())).sort_values("order", ascending=False).head(15)
            col.markdown(f"**{title}** ({len(sel)} genes)")
            col.dataframe(top[["x", "y"]].rename(columns={"x": "IBD flare", "y": "Sid"}).round(2), width="stretch")

        st.markdown(f"#### Inflammation genes in {ibd_ct} (IBD) and {sid_ct} (Sid): same or different?")
        st.markdown("Genes chosen in advance for what they mean in CD8 'killer' T cells. Orange = more active, blue = "
                    "less active (IBD: flare vs own calm samples; Sid: after vs before the chosen date).")
        code_of = dict(zip(names["name"], names["code"])) if names is not None and "code" in names else {}
        rows = []
        for program, genes in CD8_PANEL.items():
            for g in genes:
                ib = next((ibd.loc[k, "change"] for k in (g, code_of.get(g)) if k in ibd.index), np.nan)
                sd = next((sid_change.get(k) for k in (g, code_of.get(g)) if k in sid_change.index), np.nan)
                rows.append({"program": program, "gene": g, "IBD flare": ib, "Sid": sd})
        panel = pd.DataFrame(rows)
        shown = panel.dropna(subset=["IBD flare", "Sid"], how="all")
        if shown.empty:
            st.info("None of the panel genes are in both summaries yet. Re-run precompute.py and sid_pseudobulk.py.")
        else:
            labels = shown["program"].str.split(" /").str[0] + " · " + shown["gene"]
            z = shown[["IBD flare", "Sid"]].to_numpy()
            fig = go.Figure(go.Heatmap(z=z, x=["IBD flare", "Sid"], y=labels, zmid=0, colorscale=[[0, FINE], [0.5, "#f7f7f7"], [1, FLARE]],
                                       text=np.where(np.isnan(z), "not measured", np.round(z, 2).astype(str)),
                                       texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title="Direction of change in selected inflammation genes",
                              height=26 * len(shown) + 100, xaxis_title="Study and comparison",
                              yaxis_title="Gene", margin=dict(t=72, l=36))
            chart(fig)
            agree = shown.dropna()
            same_dir = int((np.sign(agree["IBD flare"]) == np.sign(agree["Sid"])).sum())
            st.markdown(f"Of {len(agree)} panel genes measured in both, **{same_dir} move the same way** and "
                        f"**{len(agree) - same_dir} move in opposite directions**.")
            by_prog = agree.assign(same=np.sign(agree["IBD flare"]) == np.sign(agree["Sid"])).groupby("program")["same"]
            st.dataframe(pd.DataFrame({"genes": by_prog.size(), "same direction": by_prog.sum()}), width="content")
        st.caption("Drug context (general knowledge, not from this data): approved cancer drugs block the checkpoint "
                   "'brakes' PD-1, CTLA-4 and LAG-3 to release killer T cells; the IBD drug vedolizumab blocks α4β7 "
                   "(ITGA4 + ITGB7) so immune cells can't enter the gut. Shared programs suggest shared drug effects "
                   "(and side effects); different programs suggest more targeted options.")

    if SD is not None:
        sid_vs_ibd()

    st.subheader("2. Inside IBD: Crohn's vs ulcerative colitis")
    default_ct = next((c for c in pb_types if "B Cell" in c and "Naive" in c), next((c for c in pb_types if "B Cell" in c), pb_types[0]))
    ct = st.selectbox("Cell type", pb_types, index=pb_types.index(default_ct), key="cmp_ct")
    cd, uc = flare_gene_changes(ct, "Crohn"), flare_gene_changes(ct, "Ulcerative")
    if cd.empty or uc.empty:
        st.info("Not enough Crohn's or ulcerative colitis patients with both kinds of sample for this cell type.")
    else:
        both = cd.join(uc, lsuffix="_cd", rsuffix="_uc", how="inner")
        both = both[both["activity_cd"] >= both["activity_cd"].quantile(0.25)]  # skip barely measured genes
        df = pd.DataFrame({"x": both["change_cd"], "y": both["change_uc"]})
        df.index.name = "gene"
        fig, rho, df = compare_scatter(df, "Crohn's", "UC", "Change during flare in Crohn's patients",
                                       "Change during flare in ulcerative colitis patients")
        c = st.columns(3)
        c[0].metric("Similarity of the two diseases' flares", f"{rho:+.2f}",
                    help=f"Spearman correlation of changes in the {df.attrs['n_moving']} genes that clearly change in at "
                         f"least one disease (+1 identical pattern, 0 unrelated, −1 opposite). All genes: "
                         f"{df.attrs['rho_all']:+.2f}.")
        c[1].metric("People compared", f"{int(cd['people'].iloc[0])} Crohn's · {int(uc['people'].iloc[0])} UC")
        c[2].metric("Genes compared", f"{len(df):,}")
        st.markdown(f"In **{ct}**, Crohn's and ulcerative colitis flares are **{similarity_words(rho)}** "
                    f"(similarity {rho:+.2f}, based on {df.attrs['basis']}).")
        chart(fig)
        st.caption("Each dot is a gene. Top-right: more active during flares in both diseases. Yellow/teal: changes "
                   "in one disease only. Purple: opposite directions.")
        cols = st.columns(3)
        for col, (title, mask, sort) in zip(cols, [
                ("Shared flare genes", df["pattern"].isin(["Up in both", "Down in both"]), (df["x"] + df["y"]).abs()),
                ("Crohn's only", df["pattern"] == "Crohn's only", df["x"].abs()),
                ("Ulcerative colitis only", df["pattern"] == "UC only", df["y"].abs())]):
            top = df[mask].assign(order=sort[mask]).sort_values("order", ascending=False).head(12)
            col.markdown(f"**{title}** ({int(mask.sum())} genes)")
            col.dataframe(top[["x", "y"]].rename(columns={"x": "Crohn's", "y": "UC"}).round(2), width="stretch")

    st.subheader("⚠️ Keep in mind")
    st.markdown("- **Sid is one person** with a different disease (bone cancer), treatment and lab kit; this is a "
                "pattern comparison, not a finding.\n"
                "- His 'change' is draws after vs before a date, not flare vs calm; treatment dates aren't in the data.\n"
                "- IBD here means Crohn's disease and ulcerative colitis; within IBD, each disease group is small.")

elif page == "Cell map":
    header("A map of every kind of immune cell")
    st.markdown(f"Each dot is one cell ({INFO['map_cells']:,} shown). Cells that behave alike sit close together, "
                "so each island is a kind of immune cell.")
    cells = D["cells_map"]
    if group != "Everyone":
        cells = cells[cells["group"] == group]
    c1, c2 = st.columns(2)
    color = c1.radio("Colour by", ["Cell family", "Cell type", "Detailed cell type", "Flare vs feeling fine"],
                     horizontal=True)
    split = c2.toggle("Show flare and feeling fine side by side", value=False)
    cells = cells.assign(state=cells["flare"].map(STATE)).reset_index(drop=True)
    cells["idx"] = cells.index
    col = {"Cell family": "family", "Cell type": "broad", "Detailed cell type": "detailed",
           "Flare vs feeling fine": "state"}[color]
    fig = px.scatter(cells, x="x", y="y", color=col, facet_col="state" if split else None, custom_data=["idx"],
                     color_discrete_map=STATE_COLORS if col == "state" else FAMILY_COLORS if col in ("family", "broad") else None,
                     color_discrete_sequence=DETAIL_PALETTE,
                     hover_data={"x": False, "y": False, "family": True, "broad": True, "detailed": True, "person": True},
                     render_mode="webgl", labels={col: ""})
    fig.update_traces(marker=dict(size=2.5, opacity=0.6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=650, legend=dict(itemsizing="constant"), dragmode="lasso")
    event = chart(fig, on_select="rerun", selection_mode=("lasso", "box"), key=f"cellmap_{col}_{split}_{group}")
    st.caption(f"Map: {INFO.get('map_source', '')}.")
    chosen = [p["customdata"][0] for p in (event.selection.points if event else []) if p.get("customdata")]
    if chosen:
        sel = cells.loc[chosen]
        flare_share, overall = (sel["state"] == "Flare").mean(), (cells["state"] == "Flare").mean()
        st.subheader("Your selection")
        kpis([("🔬", f"{len(sel):,}", "cells selected"),
              ("🔥", f"{flare_share:.0%}", f"from flare samples (all cells: {overall:.0%})"),
              ("👥", sel["person"].nunique(), "different people")])
        top = sel["detailed"].value_counts(normalize=True).head(6).mul(100).round(1).rename("% of selection")
        st.dataframe(top.reset_index().rename(columns={"detailed": "cell type"}), hide_index=True, width="stretch")
    else:
        st.markdown("<div class='tip'>✏️ Draw a lasso around any island of cells to see what's in it.</div>",
                    unsafe_allow_html=True)

elif page == "Genes":
    header("Which genes switch on or off during a flare?")
    st.markdown("Inside each kind of immune cell, we compare how active every gene is during a flare vs "
                "feeling fine, **each person against themselves** (bowel-disease patients).")
    gt = D["gene_tests"]
    ct = st.selectbox("Cell type", sorted(gt["cell_type"].unique()))
    g = gt[gt["cell_type"] == ct].copy()
    g["direction"] = np.where(g["padj"] < 0.1, np.where(g["change"] > 0, "Up in flare", "Down in flare"),
                              "No clear change")
    g["-log10 p"] = -np.log10(g["p"].clip(lower=1e-300))
    n_sig = (g["padj"] < 0.1).sum()
    st.markdown(f"**{n_sig}** genes change clearly in **{ct}** (adjusted p < 0.1).")
    fig = px.scatter(g, x="change", y="-log10 p", color="direction", hover_name="gene",
                     color_discrete_map={"Up in flare": FLARE, "Down in flare": FINE, "No clear change": GREY},
                     hover_data={"people_up": True, "people": True, "padj": ":.3g", "-log10 p": False},
                     labels={"change": "Flare − feeling-fine gene activity (log₂ difference)",
                             "-log10 p": "Statistical evidence: −log₁₀(p) (higher = stronger)"}, render_mode="webgl")
    fig.update_traces(marker=dict(size=5))
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    for i, (_, r) in enumerate(g.sort_values("p").head(6).iterrows()):
        # Label only the six strongest points; the rest remain available on hover.
        fig.add_annotation(x=r["change"], y=r["-log10 p"], text=r["gene"], showarrow=True,
                           arrowhead=0, arrowwidth=1, arrowcolor=GREY, ax=0,
                           ay=-18 if i % 2 == 0 else 18, font=dict(size=10))
    fig.update_layout(title=f"Gene activity changes during flares in {ct}", height=480, legend_title_text="")
    chart(fig)
    st.caption("Each dot is one gene in the selected cell type. Right of zero means more active during flares; left "
               "means less active. Higher dots have stronger evidence for a flare-related change. The six strongest "
               "genes are labelled; hover any dot to see its name and statistics.")
    top = g.sort_values("p").head(30)
    gene = st.selectbox("Look at one gene", top["gene"].tolist() + sorted(set(g["gene"]) - set(top["gene"])))
    pb = D["pseudobulk"]
    pb = pb[pb["cell_type"] == ct][["sample_id", gene]].merge(
        D["samples"][D["samples"]["group"] == "Bowel disease"][["sample_id", "person", "flare"]], on="sample_id")
    pairs = pb.groupby(["person", "flare"])[gene].mean().unstack("flare").dropna().reset_index()
    r = g[g["gene"] == gene].iloc[0]
    st.markdown(f"**{gene}** in {ct}: more active during flare in **{int(r['people_up'])} of {int(r['people'])}** people "
                f"(adjusted p {pval_text(r['padj'])}).")
    chart(slope_chart(pairs, "Mean gene activity (log₂ counts per million)",
                      f"{gene} activity in {ct}: flare vs feeling fine"))
    st.caption("Each faint line is one person, averaged across their samples in each state. Orange means that person's "
               "gene activity is higher during flares; blue means lower. The dark thick line is the study average.")
    with st.expander("Top 30 genes"):
        st.dataframe(top[["gene", "change", "people_up", "people", "p", "padj"]].round(4), hide_index=True)

elif page == "How we checked":
    header("How we checked")
    st.markdown("The judges' first question: **is it real?** Here is everything we did to make sure.")
    s = D["samples"]
    info = INFO
    st.subheader("✅ Junk removed")
    st.markdown(f"{info['cells_removed']:,} of {info['cells_total']:,} cells "
                f"({info['cells_removed'] / info['cells_total']:.0%}) were debris, two cells stuck together, "
                "or damaged. We left them out.")
    st.subheader("✅ Everyone compared with themselves")
    sg = s if group == "Everyone" else s[s["group"] == group]
    st.markdown(f"{sg.groupby('person')['flare'].nunique().eq(2).sum()} people ({group.lower()}) gave both flare and "
                "feeling-fine samples. Tests use only within-person differences.")
    st.subheader("✅ Lab batches are mixed")
    st.markdown("Samples were processed in batches. If one batch were all flares, a batch quirk could look like a "
                "flare effect. Every batch has both:")
    st.dataframe(pd.crosstab(s["run"], s["state"]), width="content")
    both_people = s.groupby("person")["flare"].nunique().eq(2)
    sb = s[s["person"].isin(both_people[both_people].index)]
    if group != "Everyone":
        sb = sb[sb["group"] == group]
    runs_per_person = sb.groupby("person")["run"].nunique()
    n_one = int(runs_per_person.eq(1).sum())
    st.markdown(f"**Even better:** of the {len(runs_per_person)} people with both kinds of sample "
                f"({group.lower()}), **{n_one}** had *all* their samples processed in the **same batch**. "
                "For them a batch quirk affects flare and calm samples equally, so it cancels out when we "
                "compare them. On *Flare signature*, tick *Only people whose samples were all processed in one "
                "batch* to see the result for just these people.")
    st.subheader("✅ Freezer time is similar")
    ft = s.groupby("state")["freezer_days"].median()
    st.markdown("Median days in the freezer before processing: " +
                ", ".join(f"**{k}** {v:.0f}" for k, v in ft.items()) + ".")
    if "steroid_hours" in s:
        st.subheader("✅ Is it the flare, or the steroids?")
        sg2 = s if group == "Everyone" else s[s["group"] == group]
        recent = sg2["steroid_hours"] <= 168
        share = sg2.assign(recent=recent).groupby("state")["recent"].mean().mul(100)
        st.markdown("Steroids are the usual treatment for flares, and they change the immune cells in blood. "
                    "Share of samples with **recorded** steroid use in the week before: " +
                    ", ".join(f"**{k}** {v:.0f}%" for k, v in share.items()) + ".")
        st.caption("⚠️ " + STEROID_CAVEAT + " So this check can rule out recorded steroid use, not all steroid use.")
        res_all_b, _ = paired_cell_stats(FAM, group, ())
        res_no_s, _ = paired_cell_stats(FAM, group, (), steroid_hours=168)
        res_none, _ = paired_cell_stats(FAM, group, (), steroid_hours=float("inf"))
        if len(res_all_b) and len(res_no_s):
            top_ct = res_all_b.iloc[0]["cell type"]
            rows = []
            for name, r in [("All samples", res_all_b),
                            ("Without samples within a week of recorded steroids", res_no_s),
                            ("Without any sample with recorded steroid use", res_none)]:
                row = r[r["cell type"] == top_ct]
                if len(row):
                    row = row.iloc[0]
                    rows.append({"samples used": name, "people": int(row["people"]),
                                 f"{top_ct}: people going up": int(row["people up"]),
                                 "average change (points)": round(row["change"], 1),
                                 "p (tested on its own)": round(row["p"], 3),
                                 "adjusted p (all families)": round(row["adjusted p"], 3)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.markdown("If the change holds in the lower rows, **recorded** steroid use isn't what's driving it. "
                        "On *Flare signature* you can try other windows.")

    st.subheader("✅ Corrected for testing many things at once")
    st.markdown("Testing many cell types at once means one could look 'significant' by luck. We use the standard "
                "Benjamini–Hochberg correction (the 'adjusted p'). Our B-cell hypothesis was chosen before the newest "
                "data arrived, so we also show its plain p-value as a single test.")
    st.subheader("✅ Does the headline hold in each disease?")
    res_all, pairs = paired_cell_stats(FAM, "Bowel disease", ())
    if len(res_all):
        top = res_all.iloc[0]["cell type"]
        sub = pairs[pairs["cell_type"] == top].merge(
            s.groupby("person")["disease"].first().reset_index(), on="person")
        rows = []
        for label, pat in [("Crohn's", "Crohn"), ("Ulcerative colitis", "Ulcerative")]:
            x = sub[sub["disease"].str.contains(pat)]
            if len(x):
                rows.append({"group": label, "people": len(x),
                             "same direction as overall": int(((x["Flaring"] - x["Baseline"]) *
                                                               np.sign(res_all.iloc[0]["change"]) > 0).sum())})
        st.markdown(f"Headline cell type: **{top}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("⚠️ Things to keep in mind")
    st.markdown("- Monocytes are fewer than in typical blood studies (at-home collection and freezing may affect them).\n"
                "- 5 people have other conditions (arthritis, lupus-related, fibromyalgia); pick *Other* in the menu "
                "to see whether their flares look similar.\n"
                "- About 30 people: good for a first finding, and worth confirming in ImYoo's held-out patients.")
