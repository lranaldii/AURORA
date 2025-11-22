from typing import List, Dict, Any
from aurora.utils.data_models import Scenario, Clause


class HardComplianceCritiqueAgent:
    """
    Rule-based compliance critic that approximates a "Regulation Prover".

    Responsibilities:
    1. Estimate clause coverage by comparing gold-linked clauses to those retrieved by the retrieval agent.
    2. Mark interactions as non-compliant when:
       - The dataset label indicates BREACH / HIGH_RISK / MISSING_OBLIGATION, or
       - Simple rule-based patterns detect an obvious violation.
    3. Optionally attach retrieval meta-data (confidence, fallback, retrieval failure) when provided.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal rule-based patterns
    # ------------------------------------------------------------------

    def _pattern_based_violation(self, text: str) -> bool:
        """
        Very lightweight rule-based violation detector.
        Provide an additional deterministic safety net to flag problematic cases even when labels are missing.
        """
        txt = text.lower()

        patterns = [
            # Mortgage affordability without income.
            ("mortgage" in txt and ("no income" in txt or "lost my job" in txt)),
            # Encouraging risky gambling behaviour.
            ("gambling" in txt and ("increase limit" in txt or "raise limit" in txt)),
            # Suggesting to stop paying existing debts.
            ("consolidate" in txt and "stop paying" in txt),
            # Unrealistic risk-free doubling of money.
            ("double my money" in txt and "no risk" in txt),
        ]

        return any(patterns)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def __call__(
        self,
        scenario: Scenario,
        retrieved_clauses: List[Clause],
        retrieval_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        # Clause coverage against gold annotations (if available).
        gold_ids = set(scenario.linked_clauses or [])
        retrieved_ids = {c.clause_id for c in retrieved_clauses}
        coverage = len(gold_ids & retrieved_ids) / max(1, len(gold_ids))

        # Explicit label-based non-compliance.
        explicit_non_compliant = scenario.compliance_label in {
            "BREACH",
            "HIGH_RISK",
            "MISSING_OBLIGATION",
        }

        # Rule-based non-compliance from text patterns.
        text = f"{scenario.user_message} {scenario.assistant_response}"
        pattern_non_compliant = self._pattern_based_violation(text)

        is_non_compliant = explicit_non_compliant or pattern_non_compliant

        result: Dict[str, Any] = {
            "coverage": coverage,
            "is_non_compliant": is_non_compliant,
            "explicit_label_non_compliant": explicit_non_compliant,
            "pattern_non_compliant": pattern_non_compliant,
            "gold_clause_ids": list(gold_ids),
            "retrieved_clause_ids": list(retrieved_ids),
        }

        # Attach retrieval meta-data when available so that downstream
        # components (escalation, audit chains) can reason about
        # uncertainty in the grounding step.
        if retrieval_meta is not None:
            result["retrieval_confidence"] = retrieval_meta.get("retrieval_confidence")
            result["retrieval_failed"] = retrieval_meta.get("retrieval_failed", False)
            result["used_web_fallback"] = retrieval_meta.get("used_web_fallback", False)

        return result
