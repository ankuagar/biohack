# ImYoo–Sid bridge roadmap

This tracks the work connecting ImYoo single-cell cluster programs with Sid's compact blood-draw summaries.
The bridge is exploratory: it compares shared gene programs across datasets and does not transfer cell labels or
establish treatment effects.

**Last updated:** 2026-10-18

| # | Goal | Status | Current evidence / next step |
|---|---|---|---|
| 1 | All-cluster heatmap across every ImYoo cluster and Sid cell group | 🟡 Partial | The portal has an all-cluster heatmap for the selected Sid cell group. Add a cluster × Sid-cell-group overview. |
| 2 | Before/after effect sizes for each cluster program | ✅ Implemented | The portal shows before, after, difference, draw counts, and a deterministic 95% bootstrap interval when both sides have at least two draws. |
| 3 | Null comparison with random gene sets of the same size | 🟡 Implemented in code | Rerun `sid_cluster_bridge.py` to create matched random-set scores; the portal then shows an empirical two-sided comparison. |
| 4 | Connect cluster programs directly to the IBD analysis | 🟡 Implemented in code | The portal compares ImYoo flare-minus-baseline marker changes with Sid before/after changes when at least three programs match. |
| 5 | Improve cluster interpretation | 🟡 Partial | Labels now include dominant existing annotation and representative markers. Add flare enrichment and the Sid group with the strongest match. |
| 6 | Parameter-study robustness | ⬜ Pending | Repeat program scoring across the five clustering resolutions and identify stable programs. |
| 7 | Evidence tiers | 🟡 Started | The bridge page now labels itself as a cross-dataset program match and adds a convergent-program shortlist; extend the tiers across the full portal. |
| 8 | Exportable reports | 🟡 Partial | CSV downloads now cover selected scores and effect tables. Add compact HTML/PDF export. |

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
