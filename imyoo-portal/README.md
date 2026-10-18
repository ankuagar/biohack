# ImYoo Flare Explorer

A clickable portal and AI analyst: what changes in the blood when a bowel-disease flare hits?
Built for the [SPC Bio-Data Hackathon](https://www.southparkcommons.com/bio-data-hackathon).

| File | What it does | Where it runs |
|---|---|---|
| `precompute.py` | Boils ImYoo's study file (v3, 5.6 GB) down to small summary tables in `data/` | **SageMaker only** (real data) |
| `app.py` | The portal (Streamlit) | SageMaker (real data) or laptop (made-up data) |
| `analyst.py` | The Flare Analyst: Claude (Amazon Bedrock, inside AWS) running the portal's analyses as tools | SageMaker or laptop |
| `.streamlit/config.toml` | The look (font, colours, rounded corners), hides the developer toolbar, reloads when files change | SageMaker or laptop |
| `sid_inspect.py` | Prints the structure of Sid Sijbrandij's public files (catalogue + blood single-cell file) | SageMaker |
| `sid_precompute.py` | Labels groups of Sid's blood cells (T-cell receptors + a model taught on ImYoo's cells + marker genes) and summarises his T-cell families over time | SageMaker |
| `sid_pseudobulk.py` | Gene activity per blood draw and cell family for Sid (reuses the group labels; no model re-run), for comparing IBD with cancer | SageMaker |
| `sid_cluster_bridge.py` | Scores ImYoo cluster marker programs in Sid's compact blood-draw summaries; does not transfer cell labels | SageMaker |
| `flare_classifier.py` | Trains simple flare-vs-calm models on ImYoo's samples (cell mix, gene activity, each vs own normal), tests each on people it never saw, with a shuffle test | SageMaker |
| `cluster_imyoo.py` | Finds unsupervised Leiden clusters from the single-cell expression representation, computes UMAP coordinates and marker genes, and writes compact portal tables | SageMaker only (real data) |
| `cluster_study.py` | Compares Leiden resolutions and neighbor graphs, writes cluster composition and stability tables | AWS HealthOmics container |
| `healthomics/` | AWS HealthOmics WDL, container recipes and run instructions for baseline clustering and parameter studies | AWS HealthOmics |
| `make_demo_data.py` | Makes **made-up** data in the same shape, for trying the portal anywhere | anywhere |

## Run it in SageMaker on the real data

1. **Copy ImYoo's v3 file** into your SageMaker space (JupyterLab → File → New → Terminal):
   ```bash
   aws s3 cp s3://spc-biohack26-imyoo/flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad ~/data/
   ```
   (v2 is the earlier release; v1 is dummy data.) Use a space with enough memory (ml.r5.2xlarge, 64 GB) and
   storage (20 GB or more).
2. **Put the code in place:** in JupyterLab's file panel make a folder `imyoo-portal`, open it, and use the
   **Upload** button (arrow icon) for `precompute.py`, `app.py`, `analyst.py` and `cluster_imyoo.py`. Dragging files in doesn't
   always overwrite old copies; Upload asks to overwrite. The theme file lives in a hidden folder, so create it
   from the terminal by copying `.streamlit/config.toml` into `~/imyoo-portal/.streamlit/`.
3. **Install and build the summaries** (about 2 minutes on v3). If the file names genes only by code (ENSG…),
   copy Sid's small 10x gene table to `~/data/sid/features.tsv.gz` first (see Sid's section) so genes get readable
   names; the script prints which gene-name column it used:
   ```bash
   cd ~/imyoo-portal
   pip install -q pyarrow streamlit plotly statsmodels scikit-learn boto3
   python precompute.py
   ```
   To add the unsupervised single-cell page, install the clustering dependencies and run:
   ```bash
   pip install -q scanpy leidenalg igraph
   python cluster_imyoo.py
   ```
   This uses a stored scVI/PCA representation when available, removes the same junk labels as `precompute.py`,
   and writes only compact tables to `data/`. Adjust `--resolution` to make fewer or more clusters.
4. **Start the portal:**
   ```bash
   streamlit run app.py --server.port 8501 --server.headless true \
     --server.enableCORS false --server.enableXsrfProtection false
   ```
5. **Open it:** take your JupyterLab address, which looks like
   `https://XXXX.studio.us-east-1.sagemaker.aws/jupyterlab/default/lab`,
   and replace the ending `lab...` with `proxy/8501/` (keep the `/` at the end).

If the page stays blank, stop it (Ctrl+C) and start it with
`--server.baseUrlPath /jupyterlab/default/proxy/absolute/8501`, then open `.../jupyterlab/default/proxy/absolute/8501/`.

The portal runs while that terminal tab is open and the space is running. After uploading a new `app.py` it
reloads by itself; after re-running `precompute.py` it picks up the new numbers by itself. Stop the space when
you're done (budget).

## Try it on a laptop with made-up data
```bash
pip install streamlit plotly pandas pyarrow scipy statsmodels scikit-learn boto3
python make_demo_data.py
streamlit run app.py
```
A yellow banner says "DEMO MODE: made-up numbers". The Flare Analyst needs AWS access to Amazon Bedrock.

## Pages
- **Home**: our hypothesis re-tested (with both the plain and the corrected p-value), key numbers, a guided tour.
- **The study**: every person's samples on a timeline; click a dot to open that sample's report.
- **A person's journey**: one person's immune-cell mix over time, flares shaded.
- **Flare signature**: which cells change during a flare, each person vs themselves; one line per person; click a
  bar to see it person by person. Robustness checks: leave out a lab batch, only people processed in one batch,
  leave out samples taken soon after steroids. Detail levels: cell families (subtypes added back together), main
  cell types, detailed cell types.
- **Seeing a flare coming**: can a blood score tell flare from calm, or warn early? Compares a one-size-fits-all
  score with scores that compare each person with their own normal, a "red flag" for big B-cell jumps, and a
  trained flare classifier (run `python flare_classifier.py` after `precompute.py`, ~10–25 minutes).
- **Ask the Flare Analyst**: ask in plain English; Claude runs the analyses, shows charts and the exact numbers it
  saw. "Explore on its own" runs hypothesis → test → result → next question by itself.
- **My immune report**: a patient-friendly view of one sample vs the person's own normal (how unusual each cell
  family is compared with calm days across the study, a B-cell timeline, "Explain my report" by Claude).
- **Sid's journey**: Sid Sijbrandij's openly shared osteosarcoma data. His blood collection is almost all T cells,
  so the page follows his T-cell families across ten blood draws and compares families before and after a chosen
  date, such as his tumour surgery.
- **Compare conditions**: IBD vs cancer. How genes change during IBD flares compared with how Sid's genes change
  after vs before a date (default: his tumour surgery), plus Crohn's vs ulcerative colitis. Compares *changes*, not raw
  levels, so differences between people and lab kits largely cancel out; similarity = correlation of changes. The page
  also ranks a small set of checkpoint and trafficking genes as target hypotheses, then keeps the Sid-specific B7-H3 /
  EphA2 binder hypotheses separate because blood immune data does not measure tumor-surface abundance.
- **Cell map**: every kind of immune cell as islands of dots; lasso a group to see what's in it.
- **Unsupervised clusters**: expression-based Leiden clusters, their existing cell-type composition, flare-minus-baseline
  enrichment, and top marker genes. Cluster labels are exploratory and do not replace biological annotation.
- **Cluster studies**: compares multiple resolutions and neighbor graphs, with configuration stability and cluster
  composition by flare status, person, sample, and existing cell labels.
- **Sid cluster bridge**: scores Sid's compact blood-draw summaries against the marker-gene programs found in each
  ImYoo cluster, with an all-cluster heatmap and before/after program ranking. This compares shared gene programs
  across datasets; it does not transfer ImYoo cell labels to Sid.
- **Genes**: which genes switch on or off during a flare inside each cell type.
- **How we checked**: junk removed, lab batches, freezer time, steroids, correction for testing many things, and
  whether the finding holds in Crohn's and ulcerative colitis separately.

Data rule: real data stays in SPC's AWS. People are shown as P01, P02, …

The planned ImYoo–Sid bridge work is tracked in [SID_BRIDGE_ROADMAP.md](SID_BRIDGE_ROADMAP.md), including what is
implemented and what still needs validation.

## Sid's journey (public osteosarcoma data)

Sid's bucket is public but outside the hackathon's buckets, so copy it **anonymously** (`--no-sign-request`):
```bash
mkdir -p ~/data/sid/catalog
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/catalog/ ~/data/sid/catalog/ --recursive --region us-west-2 --no-sign-request
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/hudson_lab/PBMC_scRNAseq/seurat_objects/combined_seurat_object_20260323.h5ad ~/data/sid/ --region us-west-2 --no-sign-request
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/hudson_lab/PBMC_scRNAseq/cellranger/2026-02-13/2026-02-13_capture1/outs/multi/count/raw_feature_bc_matrix/features.tsv.gz ~/data/sid/ --region us-west-2 --no-sign-request
cd ~/imyoo-portal && python sid_precompute.py
python sid_pseudobulk.py   # gene activity per draw, for 'Compare conditions' (~5 minutes)
python sid_cluster_bridge.py  # score ImYoo cluster marker programs in Sid's blood draws
```
It also needs the ImYoo file in `~/data/` (it learns cell types from it) and takes about 7 minutes. The gene table
(`features.tsv.gz`) lets it match gene names between the two files. Output goes to `data/sid/`; the portal picks it up.

The bridge reads the compact `data/cluster_markers.parquet` and `data/sid/pseudobulk.parquet` tables. It writes
`data/sid_cluster_scores.parquet`, `data/sid_cluster_bridge_summary.parquet`, and
`data/sid_cluster_bridge_info.json`; these are ignored data artifacts and should stay on the SageMaker machine.
Refresh the portal and open **🔗 Sid cluster bridge**. A score is standardized within each Sid cell group across
draws, using the shared marker genes of one ImYoo cluster. It is exploratory evidence for a related biological program, not a claim
that Sid's cells belong to that ImYoo cluster or that the program is a drug target.

**What we learned:** a model taught on ImYoo's cells wasn't reliable on Sid's cells on its own, because the two labs
used different kits. Labelling groups of similar cells, anchored on T-cell receptors, was.
