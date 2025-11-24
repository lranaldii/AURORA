from typing import List, Dict, Any
from aurora.utils.data_models import Scenario
from aurora.agents.llm_answer_judge import LLMAnswerJudge


# task types where exact match is expected
EXACT_MATCH_TASKS = {"xbrl", "mof"}
# task types where answers are textual and benefit from semantic evaluation
SEMANTIC_MATCH_TASKS = {"cdm", "reg_qa", "definition", "dialogue_safety"}


def answer_accuracy(audit_chains: List[Dict[str, Any]], 
                    scenarios: List[Scenario],
                    use_llm_judge: bool = True,
                    llm_model: str = "gpt-4o-mini") -> float:

    scenario_map = {s.scenario_id: s for s in scenarios}
    correct = 0
    total = 0

    llm_judge = LLMAnswerJudge(model=llm_model) if use_llm_judge else None

    for chain in audit_chains:
        sid = chain["scenario_id"]
        if sid not in scenario_map:
            continue
        scenario = scenario_map[sid]

        # skip if no gold standard
        if scenario.gold_answer is None:
            continue

        final_answer = chain.get("final_answer", scenario.assistant_response)

        def norm(x: str) -> str:
            return x.strip().lower()

        gold = scenario.gold_answer
        total += 1

        # -------------------------
        # Exact match tasks
        # -------------------------
        if scenario.task_type in EXACT_MATCH_TASKS:
            if norm(final_answer) == norm(gold):
                correct += 1
            continue  # skip semantic judge

        # ---------------------------------------------------
        # Semantic tasks (definition, QA, CDM, etc.)
        # ---------------------------------------------------
        if scenario.task_type in SEMANTIC_MATCH_TASKS:
            # Short definitions sometimes should be exact match
            gold_len = len(gold.split())
            pred_len = len(final_answer.split())

            # RULE: if gold answer is short → exact match
            if gold_len <= 10 and pred_len <= 10:
                if norm(final_answer) == norm(gold):
                    correct += 1
                continue

            # Otherwise: use LLM-as-Judge for semantic correctness
            if use_llm_judge and llm_judge is not None:
                try:
                    judgment = llm_judge(
                        predicted=final_answer,
                        gold=gold
                    )
                    correct += judgment  # 0 or 1
                except Exception:
                    pass  # treat as incorrect

            continue

        # ---------------------------------------------------
        # default fallback (use exact match)
        # ---------------------------------------------------
        if norm(final_answer) == norm(gold):
            correct += 1

    return correct / max(1, total)



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
