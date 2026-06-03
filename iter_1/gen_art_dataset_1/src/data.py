#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""Convert processed POCE dataset to exp_sel_data_out schema format."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(sys.stdout, level="INFO",
           format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

Path("logs").mkdir(exist_ok=True)

WORKSPACE = Path(__file__).parent


def build_input(row: dict) -> str:
    """Build the input string: premises + query."""
    premises = row["premises"].strip()
    query = row["query"].strip()
    return f"Premises:\n{premises}\n\nQuery: {query}"


def build_output(row: dict) -> str:
    """Build the output string: gold label + proof tree summary."""
    label = row["gold_label"]
    proof = row.get("proof_tree") or ""
    explicit_count = row["metadata"]["num_explicit"]
    implicit_count = row["metadata"]["num_implicit"]

    parts = [f"Label: {label}"]
    if proof:
        parts.append(f"Proof: {proof.strip()}")
    parts.append(f"Explicit predicates: {explicit_count}, Implicit predicates: {implicit_count}")
    return "\n".join(parts)


@logger.catch(reraise=True)
def main():
    data_out_path = WORKSPACE / "data_out.json"
    logger.info(f"Loading {data_out_path}")
    data = json.loads(data_out_path.read_text())

    output_datasets = []

    for split_name, split in data["splits"].items():
        rows = split["rows"]
        logger.info(f"Processing split '{split_name}': {len(rows)} rows")

        examples = []
        for row in rows:
            meta = row["metadata"]

            # Convert explicit/implicit predicates to JSON strings for metadata
            explicit_str = json.dumps(row["explicit_predicates"])
            implicit_str = json.dumps(row["implicit_predicates"])
            proof_tree_str = row.get("proof_tree") or ""

            example = {
                "input": build_input(row),
                "output": build_output(row),
                "metadata_id": row["id"],
                "metadata_benchmark": row["benchmark"],
                "metadata_task": row["task"],
                "metadata_gold_label": row["gold_label"],
                "metadata_proof_depth": row["proof_depth"],
                "metadata_annotation_status": meta["annotation_status"],
                "metadata_num_explicit": meta["num_explicit"],
                "metadata_num_implicit": meta["num_implicit"],
                "metadata_source_split": meta["source_split"],
                "metadata_explicit_predicates": explicit_str,
                "metadata_implicit_predicates": implicit_str,
                "metadata_proof_tree": proof_tree_str[:500] if proof_tree_str else "",
                "metadata_char_count_premises": meta["char_count_premises"],
            }
            examples.append(example)

        output_datasets.append({
            "dataset": split_name,
            "examples": examples,
        })

    full_output = {
        "metadata": {
            "description": data["description"],
            "version": data["version"],
            "stats": data["stats"],
        },
        "datasets": output_datasets,
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(full_output, indent=2))
    total_examples = sum(len(d["examples"]) for d in output_datasets)
    logger.info(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Total examples: {total_examples} across {len(output_datasets)} datasets")


if __name__ == "__main__":
    main()
