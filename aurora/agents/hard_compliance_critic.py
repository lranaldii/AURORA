from typing import List, Dict, Any
from aurora.utils.data_models import Scenario, Clause


class HardComplianceCritiqueAgent:
    """
    Rule-based compliance critic
    Responsibilities:
    - Detect clear non-compliant behaviours through pattern-based rules
    - *Optional compute clause coverage for evaluation
    - *Optional Attach retrieval metadata for transparency
    """

    def __init__(self, use_oracle_labels: bool = False) -> None:
        """
        use_oracle_labels:
            If True, the agent will use dataset compliance labels
            (BREACH, HIGH_RISK, etc.). This should ONLY be enabled in controlled evaluation settings
        """
        self.use_oracle_labels = use_oracle_labels

    # ------------------------------------------------------------------
    # Internal rule-based patterns
    # ------------------------------------------------------------------

    def _pattern_based_violation(self, text: str) -> bool:
        """
        Lightweight rule-based violation detector.
        Provides deterministic detection of obvious breaches.
        """
        txt = text.lower()

        patterns = [
            # Mortgage affordability without income.
            ("mortgage" in txt and ("no income" in txt or "lost my job" in txt)),

            # Encouraging gambling escalation.
            ("gambling" in txt and ("increase limit" in txt or "raise limit" in txt)),

            # Suggesting high-cost credit to vulnerable users.
            ("payday loan" in txt and ("debt" in txt or "cannot pay" in txt)),

            # Suggesting to stop paying debts.
            ("stop paying" in txt and "debt" in txt),

            # Unrealistic guarantees (“double my money with no risk”)
            ("double" in txt and "no risk" in txt),
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

        # 1. Clause coverage for evaluation purposes.
        gold_ids = set(scenario.linked_clauses or [])
        retrieved_ids = {c.clause_id for c in retrieved_clauses}
        coverage = len(gold_ids & retrieved_ids) / max(1, len(gold_ids))

        # 2. Oracle label usage (OFF by default)
        explicit_non_compliant = False
        if self.use_oracle_labels:
            explicit_non_compliant = scenario.compliance_label in {
                "BREACH",
                "HIGH_RISK",
                "MISSING_OBLIGATION",
            }

        # 3. Pattern-based deterministic rule violation
        text = f"{scenario.user_message} {scenario.assistant_response}"
        pattern_non_compliant = self._pattern_based_violation(text)

        # 4. Final boolean decision
        is_non_compliant = explicit_non_compliant or pattern_non_compliant

        # 5. Build structured result
        result: Dict[str, Any] = {
            "coverage": coverage,
            "is_non_compliant": is_non_compliant,
            "explicit_label_non_compliant": explicit_non_compliant,
            "pattern_non_compliant": pattern_non_compliant,
            "gold_clause_ids": list(gold_ids),
            "retrieved_clause_ids": list(retrieved_ids),
        }

        # 6. Attach retrieval meta-data if provided
        if retrieval_meta is not None:
            result["retrieval_confidence"] = retrieval_meta.get("retrieval_confidence")
            result["retrieval_failed"] = retrieval_meta.get("retrieval_failed", False)
            result["used_web_fallback"] = retrieval_meta.get("used_web_fallback", False)

        return result
