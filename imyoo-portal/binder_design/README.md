# Biohub binder-design bonus track

This folder prepares the public-data-only Biohub experiment for Sid's journey.
The target is human B7-H3/CD276, selected from the Sid notes because the tumour
resection was stained for CD276 and because B7-H3 has a resolved human
extracellular-domain complex structure.

## Target input

`b7h3_target.fasta` is the 432-residue extracellular target sequence used for
the design input. It is the human CD276 chain C sequence from PDB **9LY5**, with
the deposited C-terminal expression tag removed. It contains no ImYoo data and
no private Sid files.

Sources:

- Sid target-selection plan: `~/Documents/biohack26-aws-notes.md`, section 3.
- UniProt CD276/B7-H3: <https://www.uniprot.org/uniprotkb/Q5ZPR3/entry>
- Human B7-H3 structure, PDB 9LY5: <https://www.rcsb.org/structure/9LY5>

## Run the Biohub design

Biohub's current public binder-design tutorial uses the Biohub ESM package and
Modal for GPU jobs:

<https://github.com/Biohub/esm/blob/main/cookbook/tutorials/binder_design.ipynb>

The tutorial requires a Modal token and launches GPU jobs. In the notebook,
replace the custom target example with:

```python
from pathlib import Path

target_sequence = Path("binder_design/b7h3_target.fasta").read_text().splitlines()
target_sequence = "".join(line.strip() for line in target_sequence if not line.startswith(">"))

future = app.design.spawn(
    target_name="b7h3-cd276",
    target_sequence=target_sequence,
    binder_name="minibinder",
    binder_sequence="#" * 60,
    seed=0,
    batch_size=1,
)
```

The **Biohub Platform API key** is still useful for ESMFold2 or ESMC inference,
but the current public binder-design notebook does not use that key as a
replacement for Modal. Keep the key in the SageMaker environment or a secret
manager; do not paste it into this repository or into chat.

### Fold the target with the Biohub API

From the repository root, with `ESM_API_KEY` configured in the same terminal:

```bash
pip install -q esm
python binder_design/biohub_fold.py
```

This writes a public-target structure and confidence summary to
`binder_design/results/`. Once a candidate binder FASTA exists, score the
complex with:

```bash
python binder_design/biohub_fold.py --binder binder_design/candidate.fasta
```

The output includes mean pLDDT, pTM, and (for a complex) ipTM. These are model
confidence signals, not measured affinity or evidence of a working drug.

For a small demo, run several seeds (for example 0–7), collect the returned
`designed_sequence`, `iptm`, `distogram_iptm_proxy`, and the predicted complex
PDB for the best-ranked design. The final deliverable should contain 2–3
distinct sequences and their scores, plus one rendered predicted complex image.

For the first end-to-end B7-H3 job, the repository includes a launcher that
uses the official deployed Modal class:

```bash
pip install -q modal py3dmol pyarrow
modal token new
wget -q -O binder_design/modal_binder_design.py \
  https://raw.githubusercontent.com/Biohub/esm/main/cookbook/tutorials/binder_design.py
modal deploy binder_design/modal_binder_design.py
python binder_design/run_b7h3_modal.py
```

The launcher starts one 60-residue minibinder job, prints a Modal dashboard
URL, and writes candidate sequences, critic scores, and PDB files under
`binder_design/results/`. Use `biohub_fold.py --binder ...` afterward to score
the selected candidate complex through the Biohub API.

Do not upload ImYoo data or private Sid files to Biohub or Modal. The target
sequence here is public protein sequence data.

## How to present it

Say: “B7-H3/CD276 is a candidate target nominated from Sid's public tumour
record. These are computer-designed binder candidates scored by the Biohub
model. They are not drugs: binding, specificity, safety, manufacturability,
and activity have not been tested in a laboratory.”

Do not describe a candidate as a treatment for Sid or as experimentally
validated. A model score is a prioritization signal, not an affinity result.
