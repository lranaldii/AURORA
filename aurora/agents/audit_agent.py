from typing import Dict, Any
from aurora.utils.data_models import Scenario


class EscalationAgent:
    """
    Audit and decides whether to escalate an interaction to a human reviewer using hard compliance and soft risk signals.
    """

    def __init__(self, risk_threshold: float = 0.5):
        self.risk_threshold = risk_threshold

    def __call__(
        self,
        scenario: Scenario,
        hard_result: Dict[str, Any],
        soft_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        must_escalate = (
            soft_result["risk_score"] >= self.risk_threshold
            or hard_result["is_non_compliant"]
        )
        reasons = []
        if hard_result["is_non_compliant"]:
            reasons.append("Non-compliant label in annotation.")
        if soft_result["risk_score"] >= self.risk_threshold:
            reasons.append(f"Risk score {soft_result['risk_score']:.2f}.")

        return {
            "escalate": must_escalate,
            "reason": " ".join(reasons) if reasons else "No escalation trigger.",
            "suggested_priority": "HIGH" if must_escalate else "LOW",
        }
