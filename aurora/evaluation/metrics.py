from typing import List, Dict, Any
from aurora.utils.data_models import Scenario


def clause_coverage(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
) -> float:
    """
    Computes average proportion of gold clause IDs that appear in the audit chain linked_clauses
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    scores = []

    for chain in audit_chains:
        sid = chain["scenario_id"]
        if sid not in scenario_by_id:
            continue
        gold = set(scenario_by_id[sid].linked_clauses)
        predicted = {c["clause_id"] for c in chain.get("linked_clauses", [])}
        if not gold:
            continue
        scores.append(len(gold & predicted) / len(gold))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def escalation_accuracy(
    audit_chains: List[Dict[str, Any]],
    scenarios: List[Scenario],
) -> float:
    """
    Compares escalation decision against annotated escalation_required
    """
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    correct = 0
    total = 0

    for chain in audit_chains:
        sid = chain["scenario_id"]
        if sid not in scenario_by_id:
            continue
        gold = scenario_by_id[sid].escalation_required
        predicted = bool(chain.get("escalation_decision", {}).get("escalate", False))
        if gold == predicted:
            correct += 1
        total += 1

    if total == 0:
        return 0.0
    return correct / total
