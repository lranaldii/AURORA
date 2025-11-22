from typing import List, Dict, Any
from dataclasses import replace

from aurora.utils.data_models import Scenario, AuditChain

from aurora.agents.clause_retrieval import HybridRAGClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.audit_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent

from aurora.llm.base import BaseLLM


class IterativeAuroraPipeline:
    """
    Iterative self-refinement pipeline for AURORA.
    """

    def __init__(
        self,
        kb,
        llm: BaseLLM,
        top_k_clauses: int = 5,
        retrieval_threshold: float = 0.35,
        risk_threshold: float = 0.5,
        max_iterations: int = 2,
    ) -> None:

        # Shared LLM
        self.llm = llm

        # Agents
        self.retriever = HybridRAGClauseRetrievalAgent(
            kb, top_k=top_k_clauses, threshold=retrieval_threshold
        )
        self.hard_critic = HardComplianceCritiqueAgent()
        self.soft_critic = SoftRiskCritiqueAgent(llm=self.llm)
        self.escalator = EscalationAgent(risk_threshold=risk_threshold)
        self.audit_builder = AuditChainBuilderAgent(llm=self.llm)

        self.max_iterations = max_iterations

    # ---------------------------------------------------------------------

    def run(self, scenario: Scenario) -> List[AuditChain]:

        audit_trail: List[AuditChain] = []

        current_scenario = replace(scenario)
        current_response = current_scenario.assistant_response

        for step in range(self.max_iterations):

            current_scenario.assistant_response = current_response
            combined_text = f"{current_scenario.user_message} {current_scenario.assistant_response}"

            retrieval_output = self.retriever(combined_text)
            retrieved_clauses = retrieval_output["clauses"]
            retrieval_meta: Dict[str, Any] = {
                "retrieval_confidence": retrieval_output.get("retrieval_confidence"),
                "used_web_fallback": retrieval_output.get("used_web_fallback", False),
                "retrieval_failed": retrieval_output.get("retrieval_failed", False),
            }

            hard_result = self.hard_critic(
                current_scenario, retrieved_clauses, retrieval_meta
            )
            soft_result = self.soft_critic(
                current_scenario, retrieval_meta, hard_result
            )
            escalation_result = self.escalator(
                current_scenario, hard_result, soft_result, retrieval_meta
            )

            audit_chain = self.audit_builder(
                current_scenario,
                retrieved_clauses,
                hard_result,
                soft_result,
                escalation_result,
                retrieval_meta,
            )
            audit_trail.append(audit_chain)

            # ---------------- STOPPING CONDITIONS ----------------
            improvement = audit_chain.improvement_suggestion or {}
            new_response = improvement.get("replacement_guidance", "").strip()

            if step == 0:
                high_risk = soft_result.get("risk_level", "LOW") == "HIGH"
                non_compliant = hard_result.get("is_non_compliant", False)
                if (not high_risk) and (not non_compliant) and not new_response:
                    break

            if not new_response:
                break

            # New iteration starts with refined answer
            current_response = new_response

        return audit_trail
