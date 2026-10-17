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
| `make_demo_data.py` | Makes **made-up** data in the same shape, for trying the portal anywhere | anywhere |

## Run it in SageMaker on the real data

1. **Copy ImYoo's v3 file** into your SageMaker space (JupyterLab → File → New → Terminal):
   ```bash
   aws s3 cp s3://spc-biohack26-imyoo/flare-study/v3/imyoo_spc_biohackathon_20261017T211331Z.h5ad ~/data/
   ```
   (v2 is the earlier release; v1 is dummy data.) Use a space with enough memory (ml.r5.2xlarge, 64 GB) and
   storage (20 GB or more).
2. **Put the code in place:** in JupyterLab's file panel make a folder `imyoo-portal`, open it, and use the
   **Upload** button (arrow icon) for `precompute.py`, `app.py` and `analyst.py`. Dragging files in doesn't
   always overwrite old copies; Upload asks to overwrite. The theme file lives in a hidden folder, so create it
   from the terminal by copying `.streamlit/config.toml` into `~/imyoo-portal/.streamlit/`.
3. **Install and build the summaries** (about 2 minutes on v3):
   ```bash
   cd ~/imyoo-portal
   pip install -q pyarrow streamlit plotly statsmodels scikit-learn boto3
   python precompute.py
   ```
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
  score with scores that compare each person with their own normal, plus a "red flag" for big B-cell jumps.
- **Ask the Flare Analyst**: ask in plain English; Claude runs the analyses, shows charts and the exact numbers it
  saw. "Explore on its own" runs hypothesis → test → result → next question by itself.
- **My immune report**: a patient-friendly view of one sample vs the person's own normal (how unusual each cell
  family is compared with calm days across the study, a B-cell timeline, "Explain my report" by Claude).
- **Sid's journey**: Sid Sijbrandij's openly shared osteosarcoma data. His blood collection is almost all T cells,
  so the page follows his T-cell families across ten blood draws and compares families before and after a chosen
  date, such as his tumour surgery.
- **Cell map**: every kind of immune cell as islands of dots; lasso a group to see what's in it.
- **Genes**: which genes switch on or off during a flare inside each cell type.
- **How we checked**: junk removed, lab batches, freezer time, steroids, correction for testing many things, and
  whether the finding holds in Crohn's and ulcerative colitis separately.

Data rule: real data stays in SPC's AWS. People are shown as P01, P02, …

## Sid's journey (public osteosarcoma data)

Sid's bucket is public but outside the hackathon's buckets, so copy it **anonymously** (`--no-sign-request`):
```bash
mkdir -p ~/data/sid/catalog
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/catalog/ ~/data/sid/catalog/ --recursive --region us-west-2 --no-sign-request
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/hudson_lab/PBMC_scRNAseq/seurat_objects/combined_seurat_object_20260323.h5ad ~/data/sid/ --region us-west-2 --no-sign-request
aws s3 cp s3://sid-sijbrandij-osteosarc-dataset/hudson_lab/PBMC_scRNAseq/cellranger/2026-02-13/2026-02-13_capture1/outs/multi/count/raw_feature_bc_matrix/features.tsv.gz ~/data/sid/ --region us-west-2 --no-sign-request
cd ~/imyoo-portal && python sid_precompute.py
```
It also needs the ImYoo file in `~/data/` (it learns cell types from it) and takes about 7 minutes. The gene table
(`features.tsv.gz`) lets it match gene names between the two files. Output goes to `data/sid/`; the portal picks it up.

**What we learned:** a model taught on ImYoo's cells wasn't reliable on Sid's cells on its own, because the two labs
used different kits. Labelling groups of similar cells, anchored on T-cell receptors, was.
