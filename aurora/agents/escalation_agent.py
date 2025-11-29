from typing import Dict, Any
from aurora.utils.data_models import Scenario


class EscalationAgent:
    """
    Decides whether an interaction should be escalated to a human reviewer
    Escalation rules combine:
    - Hard-rule detection of compliance breaches
    - Soft LLM-based risk assessment
    - Retrieval reliability signals
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

        # ------------------------------------------------------------------
        # 1. Hard compliance violations
        # ------------------------------------------------------------------
        is_non_compliant = hard_result.get("is_non_compliant", False)
        if is_non_compliant:
            reasons.append("Hard-rule analysis indicates a likely compliance breach.")

        # ------------------------------------------------------------------
        # 2. Soft risk critic
        # ------------------------------------------------------------------
        risk_score = float(soft_result.get("risk_score", 0.0))
        risk_level = soft_result.get("risk_level", "LOW")
        if risk_score >= self.risk_threshold:
            reasons.append(
                f"Soft critic estimates {risk_level} risk (score={risk_score:.2f})."
            )

        # ------------------------------------------------------------------
        # 3. Retrieval reliability
        # ------------------------------------------------------------------
        retrieval_failed = False
        low_confidence = False

        if retrieval_meta is not None:
            retrieval_failed = retrieval_meta.get("retrieval_failed", False)
            if retrieval_failed:
                reasons.append("Clause retrieval failed or produced unreliable results.")

            conf = retrieval_meta.get("retrieval_confidence", None)
            if conf is not None and conf < 0.2:
                low_confidence = True
                reasons.append("Clause retrieval confidence is very low (<0.20).")

        # ------------------------------------------------------------------
        # 4. Escalation Rule
        # ------------------------------------------------------------------
        escalate = (
            is_non_compliant
            or risk_score >= self.risk_threshold
            or retrieval_failed
            or low_confidence
        )

        # ------------------------------------------------------------------
        # 5. Reason Handling
        # ------------------------------------------------------------------
        if escalate and not reasons:
            reasons.append("Escalation triggered due to detected risk conditions.")

        if not escalate:
            reasons.append("No strong indicators of breach or elevated risk detected.")

        return {
            "escalate": escalate,
            "reason": "; ".join(reasons),
        }
