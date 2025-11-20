"""
Run evaluation for the AURORA framework.

This script:
1. Loads the regulatory KB, scenarios, and audit chains.
2. Optionally runs the full pipeline to (re)generate audit chains.
3. Computes:
   - Clause retrieval coverage
   - Precision/Recall@k for clause retrieval
   - Escalation accuracy and P/R/F1

Usage:
    python scripts/run_evaluation.py \
        --kb data/regulatory_kb.json \
        --scenarios data/aurora_scenarios.jsonl \
        --audit_chains outputs/aurora_audit_chains.json \
        --run_pipeline

"""

import argparse
import json
from typing import List, Dict, Any

from aurora.utils.data_models import (
    Clause,
    Scenario,
    AuditChain,
    load_kb_from_json,
    load_scenarios_from_jsonl,
)
from aurora.pipeline.aurora_pipeline import run_aurora_pipeline


# -------------------------
# Utility: load audit chains
# -------------------------

def load_audit_chains(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Metrics
# -------------------------

def clause_coverage(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
) -> float:
    """
    Percentage of scenarios where at least one gold clause is present in the predicted clauses
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    covered = 0
    total = 0

    for chain in audit_chains:
        sid = chain.get("scenario_id")
        if sid not in scenario_by_id:
            continue
        gold = set(scenario_by_id[sid].linked_clauses)
        if not gold:
            continue
        total += 1
        predicted_ids = [c["clause_id"] for c in chain.get("linked_clauses", [])]
        if gold & set(predicted_ids):
            covered += 1

    return covered / total if total > 0 else 0.0


def precision_recall_at_k(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
    k: int = 5,
) -> Dict[str, float]:
    """
    Precision@k and Recall@k for clause retrieval.
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    precisions = []
    recalls = []

    for chain in audit_chains:
        sid = chain.get("scenario_id")
        if sid not in scenario_by_id:
            continue

        gold = set(scenario_by_id[sid].linked_clauses)
        if not gold:
            continue

        predicted = [
            c["clause_id"] for c in chain.get("linked_clauses", [])
        ][:k]
        pred_set = set(predicted)

        tp = len(gold & pred_set)
        if len(pred_set) > 0:
            precisions.append(tp / len(pred_set))
        else:
            precisions.append(0.0)
        recalls.append(tp / len(gold))

    return {
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
    }


def escalation_accuracy(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
) -> float:
    """
    Accuracy of escalation decisions (escalate: True/False).
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    correct = 0
    total = 0

    for chain in audit_chains:
        sid = chain.get("scenario_id")
        if sid not in scenario_by_id:
            continue
        gold = scenario_by_id[sid].escalation_required
        pred = bool(chain.get("escalation_decision", {}).get("escalate", False))
        if gold is None:
            continue
        total += 1
        if gold == pred:
            correct += 1

    return correct / total if total > 0 else 0.0


def escalation_prf(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
) -> Dict[str, float]:
    """
    Precision / Recall / F1 for the positive class: 'escalate = True'.
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}

    tp = fp = fn = 0

    for chain in audit_chains:
        sid = chain.get("scenario_id")
        if sid not in scenario_by_id:
            continue
        gold = scenario_by_id[sid].escalation_required
        if gold is None:
            continue
        pred = bool(chain.get("escalation_decision", {}).get("escalate", False))

        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=str, required=True,
                        help="Path to regulatory KB JSON.")
    parser.add_argument("--scenarios", type=str, required=True,
                        help="Path to scenarios JSONL.")
    parser.add_argument("--audit_chains", type=str, required=True,
                        help="Path to audit chains JSON.")
    parser.add_argument("--top_k_clauses", type=int, default=5)
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    parser.add_argument(
        "--run_pipeline",
        action="store_true",
        help="If set, run the AURORA pipeline to regenerate audit chains before evaluation.",
    )

    args = parser.parse_args()

    # Optionally regenerate audit chains
    if args.run_pipeline:
        print("[INFO] Running AURORA pipeline to generate audit chains...")
        run_aurora_pipeline(
            kb_path=args.kb,
            scenarios_path=args.scenarios,
            output_path=args.audit_chains,
            top_k_clauses=args.top_k_clauses,
            risk_threshold=args.risk_threshold,
        )

    # Load data
    print("[INFO] Loading KB, scenarios, and audit chains...")
    kb: List[Clause] = load_kb_from_json(args.kb)
    scenarios: List[Scenario] = load_scenarios_from_jsonl(args.scenarios)
    audit_chains: List[Dict[str, Any]] = load_audit_chains(args.audit_chains)

    # Clause-level metrics
    print("\n=== Clause Retrieval Metrics ===")
    cov = clause_coverage(audit_chains, scenarios)
    print(f"Clause coverage: {cov:.3f}")

    for k in (3, 5):
        pr = precision_recall_at_k(audit_chains, scenarios, k=k)
        print(f"P@{k}: {pr['precision_at_k']:.3f} | R@{k}: {pr['recall_at_k']:.3f}")

    # Escalation metrics
    print("\n=== Escalation Metrics ===")
    acc = escalation_accuracy(audit_chains, scenarios)
    prf = escalation_prf(audit_chains, scenarios)
    print(f"Escalation accuracy: {acc:.3f}")
    print(
        f"Escalation P: {prf['precision']:.3f} | "
        f"R: {prf['recall']:.3f} | "
        f"F1: {prf['f1']:.3f}"
    )

    print("\n[INFO] Evaluation completed.")


if __name__ == "__main__":
    main()
