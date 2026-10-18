"""Launch one official Biohub ESMFold2 minibinder design for B7-H3 via Modal.

Setup inside SageMaker:
    pip install -q esm modal py3dmol pyarrow
    modal token new
    wget -q -O binder_design/modal_binder_design.py \
      https://raw.githubusercontent.com/Biohub/esm/main/cookbook/tutorials/binder_design.py
    modal deploy binder_design/modal_binder_design.py
    python binder_design/run_b7h3_modal.py

This creates one 60-residue minibinder design job. It does not upload ImYoo,
CentAI, or private Sid data; the target FASTA is public B7-H3 sequence data.
"""
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "b7h3_target.fasta"
OUT = ROOT / "results"


def fasta_sequence(path):
    return "".join(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith(">"))


def main():
    target_sequence = fasta_sequence(TARGET)
    design_cls = modal.Cls.from_name("esmfold2-design", "ESMFold2DesignModal")
    app = design_cls(use_scaling_critics=True)
    future = app.design.spawn(
        target_name="b7h3-cd276",
        target_sequence=target_sequence,
        binder_name="minibinder",
        binder_sequence="#" * 60,
        seed=0,
        batch_size=1,
    )
    print(f"Modal call: {future.object_id}")
    print(f"Dashboard: {future.get_dashboard_url()}")
    print("Waiting for the design job to finish...")
    best_sequences, trajectory, critic_results = future.get()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b7h3_best_sequences.json").write_text(json.dumps(best_sequences, indent=2, default=str) + "\n")
    safe_results = []
    for i, result in enumerate(critic_results):
        row = {k: v for k, v in result.items() if k not in {"complex", "logits"}}
        safe_results.append(row)
        complex_obj = result.get("complex")
        if complex_obj is not None and hasattr(complex_obj, "to_pdb_string"):
            (OUT / f"b7h3_candidate_{i:03d}.pdb").write_text(complex_obj.to_pdb_string())
    (OUT / "b7h3_critic_results.json").write_text(json.dumps(safe_results, indent=2, default=str) + "\n")
    print(json.dumps(safe_results, indent=2, default=str))
    print(f"Wrote candidate outputs to {OUT}")


if __name__ == "__main__":
    main()
