from typing import List, Dict, Any
from aurora.utils.data_models import Scenario, Clause, AuditChain
from aurora.agents.clause_retrieval import HybridRAGClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.escalation_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent

class IterativeAuroraPipeline:
    """
    Full multi-agent iterative refinement pipeline.
    Unlike the single-pass pipeline, this version repeatedly applies the AURORA oversight loop, updating the assistant response after each iteration using the replacement guidance produced by the audit chain.

    Motivation:
    - Many regulatory issues persist even after the initial critique.
    - Iterative refinement allows error correction across steps.
    - This mirrors real-world audit processes, where an assistant improves outputs following reviewer feedback.
    """

    def __init__(self, kb: List[Clause],
                 top_k_clauses: int = 5,
                 risk_threshold: float = 0.5,
                 n_iterations: int = 3):
        self.retriever = HybridRAGClauseRetrievalAgent(
            kb, top_k=top_k_clauses, threshold=0.35
        )
        self.hard_critic = HardComplianceCritiqueAgent()
        self.soft_critic = SoftRiskCritiqueAgent()
        self.escalator = EscalationAgent(risk_threshold=risk_threshold)
        self.builder = AuditChainBuilderAgent()
        self.n_iterations = n_iterations

    def run(self, scenario: Scenario) -> List[AuditChain]:
        """
        Returns a list of AuditChain objects representing the
        step-by-step refinement trajectory.
        """
        audit_trail: List[AuditChain] = []
        current_response = scenario.assistant_response

        for t in range(self.n_iterations):
            # update scenario with the latest response
            scenario.assistant_response = current_response

            combined_text = scenario.user_message + " " + current_response
            retrieved = self.retriever(combined_text)
            hard = self.hard_critic(scenario, retrieved)
            soft = self.soft_critic(scenario)
            esc = self.escalator(scenario, hard, soft)

            audit_chain = self.builder(
                scenario, retrieved, hard, soft, esc
            )
            audit_trail.append(audit_chain)

            # update for next iteration if guidance exists
            suggestion = audit_chain.improvement_suggestion.get(
                "replacement_guidance", None
            )
            if suggestion:
                current_response = suggestion
            else:
                break

        return audit_trail
