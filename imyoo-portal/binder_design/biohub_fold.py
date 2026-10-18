"""Fold the public B7-H3 target or a candidate binder complex with Biohub ESMFold2.

The API token is read only from ESM_API_KEY. It is never written to output files.
Run inside SageMaker after installing the official SDK:

    pip install -q esm
    python binder_design/biohub_fold.py

To score a candidate complex, provide a FASTA containing one binder sequence:

    python binder_design/biohub_fold.py --binder binder_design/candidate.fasta
"""
import argparse
import json
import os
from pathlib import Path


DEFAULT_TARGET = Path(__file__).with_name("b7h3_target.fasta")


def read_fasta(path):
    records = {}
    name = None
    chunks = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    if not records:
        raise ValueError(f"No FASTA sequence found in {path}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    ap.add_argument("--binder", help="Optional FASTA with one candidate binder sequence")
    ap.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))
    ap.add_argument("--model", default="esmfold2-fast-2026-05")
    ap.add_argument("--loops", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    token = os.environ.get("ESM_API_KEY")
    if not token:
        raise SystemExit("ESM_API_KEY is not set in this terminal")

    from esm.sdk.api import FoldingConfig
    from esm.sdk.forge import SequenceStructureForgeInferenceClient
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    target_records = read_fasta(args.target)
    target_name, target_sequence = next(iter(target_records.items()))
    chains = [ProteinInput(id="A", sequence=target_sequence)]
    labels = {"target": target_name, "target_length": len(target_sequence)}
    if args.binder:
        binder_records = read_fasta(args.binder)
        binder_name, binder_sequence = next(iter(binder_records.items()))
        chains.append(ProteinInput(id="B", sequence=binder_sequence))
        labels.update({"binder": binder_name, "binder_length": len(binder_sequence)})

    client = SequenceStructureForgeInferenceClient(model=args.model, url="https://biohub.ai", token=token)
    prediction_input = StructurePredictionInput(sequences=chains)
    try:
        config = FoldingConfig(num_loops=args.loops, num_sampling_steps=args.steps, seed=args.seed)
    except TypeError:  # older SDK builds may not expose seed in FoldingConfig
        config = FoldingConfig(num_loops=args.loops, num_sampling_steps=args.steps)
    result = client.fold_all_atom(prediction_input, config=config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "b7h3_binder_complex" if args.binder else "b7h3_target"
    (out / f"{prefix}.cif").write_text(result.complex.to_mmcif())
    def scalar(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    plddt = getattr(result, "plddt", None)
    metrics = {
        **labels,
        "model": args.model,
        "seed": args.seed,
        "mean_plddt": scalar(plddt.mean()) if plddt is not None else None,
        "ptm": scalar(getattr(result, "ptm", None)),
        "iptm": scalar(getattr(result, "iptm", None)),
        "input_is_public_sequence": True,
        "interpretation": "Model-predicted structure/confidence; not experimental binding evidence.",
    }
    (out / f"{prefix}_scores.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out / f'{prefix}.cif'}")


if __name__ == "__main__":
    main()
