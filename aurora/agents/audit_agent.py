from typing import Dict, Any
from aurora.utils.data_models import Scenario


class EscalationAgent:
    """
    Decides whether an interaction should be escalated to a human reviewer.

    The decision is based on:
    - Hard-rule non-compliance flags
    - Soft risk score
    - Retrieval failures or very low-confidence grounding

    The output is a dictionary with:
    - "escalate": bool
    - "reason": short natural-language explanation
    """

    def __init__(self, risk_threshold: float = 0.5) -> None:
        self.risk_threshold = risk_threshold

    def __call__(
        self,
        scenario: Scenario,
        hard_result: Dict[str, Any],
        soft_result: Dict[str, Any],
        retrieval_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        reasons = []

        is_non_compliant = hard_result.get("is_non_compliant", False)
        if is_non_compliant:
            reasons.append("Hard-rule analysis suggests a potential breach.")

        risk_score = float(soft_result.get("risk_score", 0.0))
        risk_level = soft_result.get("risk_level", "LOW")
        if risk_score >= self.risk_threshold:
            reasons.append(
                f"Soft risk critic estimates {risk_level} risk (score={risk_score:.2f})."
            )

        retrieval_failed = False
        if retrieval_meta is not None:
            retrieval_failed = retrieval_meta.get("retrieval_failed", False)
            if retrieval_failed:
                reasons.append("Clause retrieval is low-confidence or failed.")
            elif retrieval_meta.get("retrieval_confidence") is not None:
                conf = retrieval_meta["retrieval_confidence"]
                if conf < 0.2:
                    reasons.append("Clause retrieval confidence is very low.")

        # Escalate if any of the conditions hold, or if the scenario is
        # explicitly labelled as requiring escalation in the annotations.
        escalate = (
            is_non_compliant
            or risk_score >= self.risk_threshold
            or retrieval_failed
            or bool(getattr(scenario, "escalation_required", False))
        )

        if escalate and not reasons:
            reasons.append("Escalation triggered by annotation or configuration.")

        if not escalate:
            reasons.append("No strong indicators of breach or elevated risk detected.")

        return {
            "escalate": escalate,
            "reason": " ".join(reasons),
        }
