# ImYoo–Sid bridge roadmap

This tracks the work connecting ImYoo single-cell cluster programs with Sid's compact blood-draw summaries.
The bridge is exploratory: it compares shared gene programs across datasets and does not transfer cell labels or
establish treatment effects.

**Last updated:** 2026-10-18

| # | Goal | Status | Current evidence / next step |
|---|---|---|---|
| 1 | All-cluster heatmap across every ImYoo cluster and Sid cell group | 🟡 Partial | The portal has an all-cluster heatmap for the selected Sid cell group. Add a cluster × Sid-cell-group overview. |
| 2 | Before/after effect sizes for each cluster program | 🟡 Partial | The portal shows before, after, difference, and draw counts. Add bootstrap confidence intervals and minimum-draw warnings. |
| 3 | Null comparison with random gene sets of the same size | ⬜ Pending | Generate reproducible matched-size controls and show the real-program percentile or empirical p-value. |
| 4 | Connect cluster programs directly to the IBD analysis | ⬜ Pending | Compare ImYoo flare-minus-baseline changes with Sid before/after changes and report a gene-program correlation. |
| 5 | Improve cluster interpretation | 🟡 Partial | Labels now include dominant existing annotation and representative markers. Add flare enrichment and the Sid group with the strongest match. |
| 6 | Parameter-study robustness | ⬜ Pending | Repeat program scoring across the five clustering resolutions and identify stable programs. |
| 7 | Evidence tiers | ⬜ Pending | Distinguish observed association, cross-dataset program match, biological hypothesis, and computer-designed binder candidate. |
| 8 | Exportable reports | ⬜ Pending | Add CSV and compact HTML/PDF export for a selected cluster–Sid comparison. |

## Recommended order

1. Add bootstrap intervals and minimum-draw warnings to the existing effect-size table.
2. Add reproducible random-gene-set null controls.
3. Add the ImYoo flare-versus-Sid change correlation.
4. Test stability across the clustering parameter study.
5. Add evidence tiers and exportable reports.

## Completed bridge components

- `sid_cluster_bridge.py` translates numeric ImYoo H5AD marker IDs when needed.
- Scores are standardized within each Sid cell group across draws.
- The portal shows a selected-program timeline, event markers, matched genes, an all-cluster heatmap, and a
  before/after ranking table.
- No raw ImYoo or Sid datasets are checked into Git; generated outputs remain machine-local and ignored.
