# ImYoo Flare Explorer

A clickable portal: what changes in the blood when a bowel-disease flare hits?

| File | What it does | Where it runs |
|---|---|---|
| `precompute.py` | Boils the 4.8 GB ImYoo file down to small summary tables in `data/` | **SageMaker only** (real data) |
| `app.py` | The portal (Streamlit) | SageMaker (real) or laptop (demo) |
| `analyst.py` | The Flare Analyst: Claude (Amazon Bedrock, inside AWS) running the portal's analyses as tools | SageMaker or laptop |
| `.streamlit/config.toml` | Theme (Inter font, flare-orange accent, rounded corners), hides developer toolbar, auto-reload on file change | SageMaker or laptop |
| `make_demo_data.py` | Makes **made-up** data in the same shape, for trying the portal anywhere | anywhere |

## Run it in SageMaker on the real data

1. Open JupyterLab in the `imyoo` space. In the left file panel, make a folder `imyoo-portal` and drag
   `precompute.py`, `app.py` and `analyst.py` into it (not the `data` folder: that's demo data).
2. Open a terminal (File → New → Terminal):
   ```bash
   cd ~/imyoo-portal
   pip install -q pyarrow streamlit plotly statsmodels scikit-learn boto3
   python precompute.py
   ```
   About 10–20 minutes. It prints progress and lists the files it made (all small).
3. Start the portal:
   ```bash
   streamlit run app.py --server.port 8501 --server.headless true \
     --server.enableCORS false --server.enableXsrfProtection false
   ```
4. Open it: take your JupyterLab address, which looks like
   `https://XXXX.studio.us-east-1.sagemaker.aws/jupyterlab/default/lab`,
   and replace the ending `lab...` with `proxy/8501/`:
   `https://XXXX.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/`
   Keep the `/` at the end.

If the page loads but stays blank or spinning, stop it (Ctrl+C) and use the "absolute" proxy instead:
```bash
streamlit run app.py --server.port 8501 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --server.baseUrlPath /jupyterlab/default/proxy/absolute/8501
```
then open `https://XXXX.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/absolute/8501/`.

The portal keeps running while the terminal tab is open and the space is running.
Stop it with Ctrl+C. Stop the space when you're done (budget!).

## Try it on a laptop with made-up data
```bash
pip install streamlit plotly pandas pyarrow scipy statsmodels
python make_demo_data.py
streamlit run app.py
```

## Pages
- **Home**: the headline finding, computed live.
- **The study**: every person's samples on a timeline (flare vs feeling fine).
- **A person's journey**: one person's immune-cell mix over time, flares shaded.
- **Flare signature**: which cell types change, each person vs themselves; one line per person;
  leave out lab batches to check the result holds.
- **Cell map**: every kind of immune cell as islands of dots; colour by flare.
- **Genes**: which genes switch on/off during a flare inside each cell type; one line per person.
- **Ask the Flare Analyst**: ask questions in plain English; Claude runs the analyses, shows charts, explains.
  "Explore on its own" runs hypothesis → test → result → next question by itself.
- **My immune report**: patient-friendly view of one sample vs the person's own normal (status card, usual-range
  bands, B-cell timeline with red-flag line, "Explain my report" by Claude).
- **How we checked**: junk removed, batches, freezer time, multiple-testing correction, per-disease check.

Data rule: real data stays in SPC's AWS. People are shown as P01–P29.
