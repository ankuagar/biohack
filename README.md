# biohack

Our project for the **[SPC Bio-Data Hackathon](https://www.southparkcommons.com/bio-data-hackathon)**
(South Park Commons, October 2026): the **ImYoo Flare Explorer**, a clickable portal and AI analyst that asks
**what changes in the blood when a bowel-disease flare hits?**

## The idea

People with Crohn's disease and ulcerative colitis live with **flares**: weeks when symptoms suddenly get much
worse. ImYoo ran a study where patients collected blood **at home**, both during flares and when they felt
fine, and measured it cell by cell. Because the same people gave both kinds of sample, we can compare **each
person with their own normal**, which is the most convincing kind of comparison.

## What we found

- **B cells (antibody-making immune cells) rise during flares** in most bowel-disease patients, when each
  person's flare blood is compared with their own calm blood.
- We spotted this on ImYoo's first data release, wrote it down as our hypothesis, and **re-tested it on the
  larger release**, where it held up.
- It **isn't explained by steroids** (the usual flare treatment): it holds without samples taken soon after
  steroids.
- Honest limits: a small group of patients; the larger release includes the earlier samples; we can't yet see
  flares *coming*. Next step: ImYoo's held-out patients.

Detailed results are shown in the portal and in our hackathon presentation.

## What's in this repository

| Folder / file | What it is |
|---|---|
| `imyoo-portal/` | The portal, the AI analyst and the scripts that prepare data for them. **Start with [`imyoo-portal/README.md`](imyoo-portal/README.md)**, which explains every file and how to run it. |
| `.gitignore` | Keeps data files out of the repository (see below). |

The portal has 12 pages, including:
- **Flare signature**: which immune cells change during a flare, one line per person, with checks for lab
  batches and steroids
- **Ask the Flare Analyst**: Claude, running inside AWS, runs the analyses itself (hypothesis, test, result,
  next question) and explains them in plain words
- **My immune report**: a patient-friendly view of one blood sample compared with that person's own normal
- **Sid's journey**: the same tools applied to Sid Sijbrandij's openly shared cancer data, following his
  T-cell families across treatment
- **How we checked**: everything we did to make sure the finding is real

## Data rules

- ImYoo's data is private and **stays inside SPC's AWS environment**. All analysis runs there (SageMaker), and
  Claude is used through Amazon Bedrock inside the same environment.
- **No data files are in this repository**: `.gitignore` blocks data folders and data file types. The scripts
  boil the data down to small summaries inside AWS.
- People appear only as P01, P02, …; no attempt is made to identify anyone.
- To try the portal anywhere, `imyoo-portal/make_demo_data.py` creates **made-up** data in the same shape.
