from typing import List, Dict, Any
from dataclasses import replace

from aurora.utils.data_models import Scenario, AuditChain
from aurora.agents.clause_retrieval import HybridRAGClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.audit_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent


class IterativeAuroraPipeline:
    """
    Iterative self-refinement pipeline for AURORA

    Given an initial scenario (user message + assistant response), the pipeline can:
    1. Run the full retrieval–critique–escalation–audit chain.
    2. Use the audit chain's improvement suggestion to rewrite the assistant response.
    3. Re-run the pipeline on the refined response for a small number of iterations.

    This allows us to study whether multi-agent oversight can systematically improve model behaviour across iterations.
    """

    def __init__(
        self,
        retriever: HybridRAGClauseRetrievalAgent,
        hard_critic: HardComplianceCritiqueAgent,
        soft_critic: SoftRiskCritiqueAgent,
        escalator: EscalationAgent,
        audit_builder: AuditChainBuilderAgent,
        max_iterations: int = 2,
    ) -> None:
        self.retriever = retriever
        self.hard_critic = hard_critic
        self.soft_critic = soft_critic
        self.escalator = escalator
        self.audit_builder = audit_builder
        self.max_iterations = max_iterations

    def run(self, scenario: Scenario) -> List[AuditChain]:
        """
        Run iterative refinement on a single scenario.

        Returns the list of AuditChain objects, one per iteration.
        The first element corresponds to the original assistant response; subsequent elements correspond to refined responses.
        """
        audit_trail: List[AuditChain] = []

        # Keep the original scenario intact by working on copies.
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
                current_scenario,
                retrieved_clauses,
                retrieval_meta,
            )
            soft_result = self.soft_critic(
                current_scenario,
                retrieval_meta,
                hard_result,
            )
            escalation_result = self.escalator(
                current_scenario,
                hard_result,
                soft_result,
                retrieval_meta,
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

            # Stopping criteria:
            # - If there is no improvement suggestion, stop.
            # - If the first iteration is already compliant and low risk, stop.
            improvement = audit_chain.improvement_suggestion or {}
            new_response = improvement.get("replacement_guidance", "").strip()

            if step == 0:
                high_risk = soft_result.get("risk_level", "LOW") == "HIGH"
                non_compliant = hard_result.get("is_non_compliant", False)
                if (not high_risk) and (not non_compliant) and not new_response:
                    break

            if not new_response:
                break

            # Prepare for the next iteration with the refined answer.
            current_response = new_response

        return audit_trail
