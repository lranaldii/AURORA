from typing import List, Dict, Any
from aurora.utils.data_models import Scenario, Clause


class HardComplianceCritiqueAgent:
    """
    'hard' critic standing in for a Regulation Prover.
    1. Checks whether the clauses linked in the annotation are retrieved.
    2. Marks interactions with BREACH/HIGH_RISK/MISSING_OBLIGATION as non-compliant.
    v2.0 change with rule-based/symbolic engine that encodes obligations explicitly
    """

    def __init__(self):
        pass

    def __call__(self, scenario: Scenario, retrieved_clauses: List[Clause]) -> Dict[str, Any]:

        # --------------------------------------------------------
        # --- canonic answer task ---
        # --------------------------------------------------------
        if scenario.task_type in {"definition", "reg_qa", "xbrl", "cdm", "mof"}:

            # Se non c'è una gold answer, fai fallback al comportamento standard
            if scenario.gold_answer is None:
                pass
            else:
                # normalizzazione
                def norm(x):
                    return x.strip().lower()

                is_correct = norm(scenario.assistant_response) == norm(scenario.gold_answer)

                return {
                    "evaluation_type": "answer_accuracy",
                    "is_non_compliant": not is_correct,
                    "coverage": 1.0,  # irrilevante per questi task
                    "violated_clauses": [],
                    "explanation": (
                        "Assistant answer does not match gold answer"
                        if not is_correct
                        else "Answer matches gold answer"
                    ),
                }

        # --------------------------------------------------------
        # --- DIALOGUE TASKS ---
        # --------------------------------------------------------

        retrieved_ids = {c.clause_id for c in retrieved_clauses}
        gold_ids = set(scenario.linked_clauses)
        coverage = len(gold_ids & retrieved_ids) / max(1, len(gold_ids))

        is_non_compliant = scenario.compliance_label in {
            "BREACH",
            "HIGH_RISK",
            "MISSING_OBLIGATION",
        }

        return {
            "coverage": coverage,
            "is_non_compliant": is_non_compliant,
            "gold_clause_ids": list(gold_ids),
            "retrieved_clause_ids": list(retrieved_ids),
        }
