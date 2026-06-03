#!/usr/bin/env python3
"""Build POCE benchmark datasets from EntailmentBank and ProofWriter."""

import json
import re
import sys
import ast
import resource
from pathlib import Path
from loguru import logger

logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(sys.stdout, level="INFO",
           format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}")
logger.add("logs/build_dataset.log", rotation="30 MB", level="DEBUG")

Path("logs").mkdir(exist_ok=True)

# Memory limit: 4GB (safe on 17GB available)
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


# ─── EntailmentBank proof parser ─────────────────────────────────────────────

def parse_eb_proof(proof_str: str) -> tuple[list[str], list[str]]:
    """Parse EntailmentBank proof string into explicit leaf IDs and implicit intermediate IDs."""
    if not proof_str or not proof_str.strip():
        return [], []

    # proof format: "sent1 & sent3 -> int1: text; int1 & sent2 -> hypothesis;"
    leaf_ids: set[str] = set()
    int_texts: list[str] = []

    steps = [s.strip() for s in proof_str.split(";") if s.strip()]
    for step in steps:
        if "->" not in step:
            continue
        lhs, rhs = step.split("->", 1)
        # Extract sent/rule IDs from lhs
        tokens = re.findall(r'\b(sent\d+|rule\d+)\b', lhs)
        leaf_ids.update(tokens)
        # Extract intermediate node text from rhs (intN: text)
        rhs = rhs.strip()
        int_match = re.match(r'int\d+\s*:\s*(.+)', rhs)
        if int_match:
            int_texts.append(int_match.group(1).strip())

    return sorted(leaf_ids), int_texts


def process_eb_row(
    row: dict,
    benchmark: str,
    task: str,
    task1_proof_map: dict | None = None,
) -> dict:
    """Convert an EntailmentBank slots row to unified schema."""
    meta = row.get("meta", {}) or {}
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except Exception:
            meta = {}

    triples = meta.get("triples", {}) or {}
    proof_str = (row.get("proof") or "").strip()
    proof_depth = row.get("depth_of_proof")

    # For task_3: cross-reference with task_1 proof
    if not proof_str and task1_proof_map and row.get("id") in task1_proof_map:
        t1 = task1_proof_map[row["id"]]
        proof_str = (t1.get("proof") or "").strip()
        proof_depth = t1.get("depth_of_proof")

    try:
        leaf_ids, int_texts = parse_eb_proof(proof_str)
        explicit_preds = [triples[sid] for sid in leaf_ids if sid in triples]

        # For task_3: implicit predicates = proof leaves NOT in context
        if task == "task_3" and task1_proof_map:
            context_sents = set(triples.values())
            # Sentences needed by proof but not in given context
            t3_triples = meta.get("triples", {})
            t1_row = task1_proof_map.get(row.get("id"), {})
            t1_meta = t1_row.get("meta", {}) or {}
            if isinstance(t1_meta, str):
                try:
                    t1_meta = ast.literal_eval(t1_meta)
                except Exception:
                    t1_meta = {}
            t1_triples = t1_meta.get("triples", {}) or {}
            # Implicit = sentences in t1 proof leaves that are in t1 triples but NOT in t3 triples
            implicit_preds = [
                t1_triples[sid]
                for sid in leaf_ids
                if sid in t1_triples and t1_triples[sid] not in set(t3_triples.values())
            ]
        else:
            implicit_preds = int_texts

        annotation_status = "ok" if proof_str else "missing_proof"
    except Exception as e:
        logger.debug(f"Proof parse error for {row.get('id')}: {e}")
        explicit_preds, implicit_preds = [], []
        annotation_status = "parse_error"

    try:
        depth_int = int(proof_depth) if proof_depth else None
    except (TypeError, ValueError):
        depth_int = None

    # Gold label: task_2/task_3 are all "true" (hypothesis is proven)
    gold_label = "true"

    return {
        "id": f"eb_{task}_{row.get('source_split','train')}_{row.get('id','unk')}",
        "benchmark": "entailmentbank",
        "task": task,
        "premises": (row.get("context") or "").strip(),
        "query": (row.get("hypothesis") or row.get("question") or "").strip(),
        "gold_label": gold_label,
        "proof_tree": proof_str or None,
        "explicit_predicates": explicit_preds,
        "implicit_predicates": implicit_preds,
        "proof_depth": depth_int,
        "metadata": {
            "source_split": row.get("source_split", "train"),
            "annotation_status": annotation_status,
            "num_explicit": len(explicit_preds),
            "num_implicit": len(implicit_preds),
            "num_distractors": 0,
            "char_count_premises": len((row.get("context") or "")),
        },
    }


# ─── ProofWriter proof parser ─────────────────────────────────────────────────

def parse_pw_proof(proof_str: str, theory: str) -> tuple[list[str], list[str]]:
    """Parse ProofWriter allProofs field into explicit/implicit predicates."""
    if not proof_str or not proof_str.strip():
        return [], []

    # Build theory sentence map: "The bear is big." → text
    # ProofWriter theory is a flat string of facts + rules
    theory_sents = [s.strip() for s in theory.split(".") if s.strip()]

    # allProofs format: "@0: The bear is big.[(triple1 OR ...)] ..."
    # Multiple proof paths start with @0, @1...
    # We use @0 (the first/primary proof)
    primary = proof_str.split("@")[1] if "@" in proof_str else proof_str
    primary = re.sub(r'@\d+:.*', '', primary, flags=re.DOTALL).strip()

    explicit_preds: list[str] = []
    implicit_preds: list[str] = []

    # Extract all facts: "Fact text.[(triple1 OR ...)]"
    fact_pattern = re.compile(r'([A-Z][^.]+\.)\s*\[([^\]]+)\]')
    for m in fact_pattern.finditer(primary):
        fact_text = m.group(1).strip()
        derivation = m.group(2).strip()
        # If derivation is purely triple references → explicit
        if re.match(r'^[\(\)tripleTOR \d]+$', derivation.replace("triple", "t")):
            explicit_preds.append(fact_text)
        else:
            # Derived via rules → implicit intermediate
            implicit_preds.append(fact_text)

    return explicit_preds, implicit_preds


def process_pw_row(row: dict, task: str, idx: int) -> dict:
    """Convert a ProofWriter row to unified schema."""
    theory = (row.get("theory") or "").strip()
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    proof_str = (row.get("allProofs") or "").strip()

    try:
        explicit_preds, implicit_preds = parse_pw_proof(proof_str, theory)
        # Filter empty
        explicit_preds = [p for p in explicit_preds if p]
        implicit_preds = [p for p in implicit_preds if p]
        annotation_status = "ok" if proof_str else "missing_proof"
    except Exception as e:
        logger.debug(f"PW proof parse error for {row.get('id')}: {e}")
        explicit_preds, implicit_preds = [], []
        annotation_status = "parse_error"

    try:
        depth_int = int(row.get("maxD") or 0)
    except (TypeError, ValueError):
        depth_int = None

    gold_label = answer.lower().strip() if answer else "unknown"
    if gold_label not in ("true", "false", "unknown"):
        gold_label = "unknown"

    return {
        "id": f"pw_{task}_{row.get('id', f'row_{idx}')}",
        "benchmark": "proofwriter",
        "task": task,
        "premises": theory,
        "query": question,
        "gold_label": gold_label,
        "proof_tree": proof_str or None,
        "explicit_predicates": explicit_preds,
        "implicit_predicates": implicit_preds,
        "proof_depth": depth_int,
        "metadata": {
            "source_split": row.get("source_split", "train"),
            "annotation_status": annotation_status,
            "num_explicit": len(explicit_preds),
            "num_implicit": len(implicit_preds),
            "num_distractors": 0,
            "char_count_premises": len(theory),
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    data_dir = Path("temp/datasets")

    # Load EntailmentBank
    logger.info("Loading EntailmentBank data...")
    eb_task1_rows = json.loads((data_dir / "raw_entailmentbank_task_1_slots.json").read_text())
    eb_task2_rows = json.loads((data_dir / "raw_entailmentbank_task_2_slots.json").read_text())
    eb_task3_rows = json.loads((data_dir / "raw_entailmentbank_task_3_slots.json").read_text())

    # Build task_1 proof map for task_3 cross-reference
    task1_proof_map = {r["id"]: r for r in eb_task1_rows}
    logger.info(f"Task1 proof map: {len(task1_proof_map)} entries")

    # Process EntailmentBank task_2
    logger.info("Processing EntailmentBank task_2...")
    eb_t2_processed = []
    for row in eb_task2_rows:
        try:
            processed = process_eb_row(row, "entailmentbank", "task_2")
            eb_t2_processed.append(processed)
        except Exception:
            logger.error(f"Failed on EB task_2 row {row.get('id')}")

    # Process EntailmentBank task_3
    logger.info("Processing EntailmentBank task_3...")
    eb_t3_processed = []
    for row in eb_task3_rows:
        try:
            processed = process_eb_row(row, "entailmentbank", "task_3", task1_proof_map)
            eb_t3_processed.append(processed)
        except Exception:
            logger.error(f"Failed on EB task_3 row {row.get('id')}")

    logger.info(f"EB task_2: {len(eb_t2_processed)} rows")
    logger.info(f"EB task_3: {len(eb_t3_processed)} rows")

    # Load ProofWriter
    pw_d3_path = data_dir / "raw_proofwriter_depth3.json"
    pw_d5_path = data_dir / "raw_proofwriter_depth5.json"

    pw_d3_processed = []
    pw_d5_processed = []

    if pw_d3_path.exists():
        logger.info("Processing ProofWriter depth-3...")
        pw_d3_rows = json.loads(pw_d3_path.read_text())
        for i, row in enumerate(pw_d3_rows):
            try:
                processed = process_pw_row(row, "depth_3", i)
                pw_d3_processed.append(processed)
            except Exception:
                logger.error(f"Failed on PW depth3 row {i}")
        logger.info(f"PW depth-3: {len(pw_d3_processed)} rows")
    else:
        logger.warning("ProofWriter depth-3 file not found yet")

    if pw_d5_path.exists():
        logger.info("Processing ProofWriter depth-5...")
        pw_d5_rows = json.loads(pw_d5_path.read_text())
        for i, row in enumerate(pw_d5_rows):
            try:
                processed = process_pw_row(row, "depth_5", i)
                pw_d5_processed.append(processed)
            except Exception:
                logger.error(f"Failed on PW depth5 row {i}")
        logger.info(f"PW depth-5: {len(pw_d5_processed)} rows")
    else:
        logger.warning("ProofWriter depth-5 file not found yet")

    # ─── Sort: ok + non-empty implicit first, then by proof_depth desc ───
    def sort_key(row: dict) -> tuple:
        status_order = {"ok": 0, "missing_proof": 1, "parse_error": 2}
        s = status_order.get(row["metadata"]["annotation_status"], 3)
        has_implicit = 0 if row["metadata"]["num_implicit"] > 0 else 1
        depth = -(row["proof_depth"] or 0)
        return (s, has_implicit, depth)

    for lst in [eb_t2_processed, eb_t3_processed, pw_d3_processed, pw_d5_processed]:
        lst.sort(key=sort_key)

    # ─── Cap to 400 examples per split ───
    eb_t2_final = eb_t2_processed[:400]
    eb_t3_final = eb_t3_processed[:400]
    pw_d3_final = pw_d3_processed[:400]
    pw_d5_final = pw_d5_processed[:400]

    # ─── Global stats ───
    all_rows = eb_t2_final + eb_t3_final + pw_d3_final + pw_d5_final
    total = len(all_rows)
    with_implicit = sum(1 for r in all_rows if r["metadata"]["num_implicit"] > 0)
    avg_implicit = sum(r["metadata"]["num_implicit"] for r in all_rows) / total if total else 0
    avg_depth = (
        sum(r["proof_depth"] for r in all_rows if r["proof_depth"]) /
        max(1, sum(1 for r in all_rows if r["proof_depth"]))
    )

    output = {
        "version": "1.0",
        "description": "POCE benchmark datasets: EntailmentBank T2/T3 + ProofWriter depth≥3",
        "splits": {
            "entailmentbank_task2": {"count": len(eb_t2_final), "rows": eb_t2_final},
            "entailmentbank_task3": {"count": len(eb_t3_final), "rows": eb_t3_final},
            "proofwriter_depth3": {"count": len(pw_d3_final), "rows": pw_d3_final},
            "proofwriter_depth5": {"count": len(pw_d5_final), "rows": pw_d5_final},
        },
        "stats": {
            "total_examples": total,
            "examples_with_implicit_predicates": with_implicit,
            "avg_implicit_per_example": round(avg_implicit, 3),
            "avg_proof_depth": round(avg_depth, 3),
        },
    }

    out_file = Path("data_out.json")
    out_file.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved data_out.json ({out_file.stat().st_size / 1e6:.1f} MB)")

    # Mini: 50 per split = 200 total
    mini_splits = {}
    for split_name, split_data in [
        ("entailmentbank_task2", eb_t2_final),
        ("entailmentbank_task3", eb_t3_final),
        ("proofwriter_depth3", pw_d3_final),
        ("proofwriter_depth5", pw_d5_final),
    ]:
        mini_splits[split_name] = {"count": min(50, len(split_data)), "rows": split_data[:50]}

    mini_output = {
        "version": "1.0",
        "description": "POCE benchmark datasets MINI (50 examples per split)",
        "splits": mini_splits,
        "stats": output["stats"],
    }
    mini_file = Path("data_out_mini.json")
    mini_file.write_text(json.dumps(mini_output, indent=2))
    logger.info(f"Saved data_out_mini.json ({mini_file.stat().st_size / 1e6:.2f} MB)")

    # Summary
    logger.info("=== FINAL STATS ===")
    logger.info(f"Total examples: {total}")
    logger.info(f"With implicit predicates: {with_implicit}/{total} ({100*with_implicit//max(1,total)}%)")
    logger.info(f"Avg implicit per example: {avg_implicit:.2f}")
    logger.info(f"Avg proof depth: {avg_depth:.2f}")
    for split_name, split_data in output["splits"].items():
        logger.info(f"  {split_name}: {split_data['count']} rows")


if __name__ == "__main__":
    main()
