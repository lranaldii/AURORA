from typing import List, Dict, Any
from aurora.utils.data_models import (
    Clause,
    Scenario,
    AuditChain,
    load_kb_from_json,
    load_scenarios_from_jsonl,
    audit_chain_to_dict,
)
from aurora.utils.json_tools import save_json
from aurora.agents.clause_retrieval import ClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.escalation_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent


def run_aurora_pipeline(
    kb_path: str,
    scenarios_path: str,
    output_path: str,
    top_k_clauses: int = 5,
    risk_threshold: float = 0.5,
) -> None:
    """
    Runs the full AURORA pipeline on all scenarios.
    1. loads the regulatory KB and scenarios
    2. runs all agents
    3. saves audit chains to JSON
    """
    kb: List[Clause] = load_kb_from_json(kb_path)
    scenarios: List[Scenario] = load_scenarios_from_jsonl(scenarios_path)

    retriever = ClauseRetrievalAgent(kb, top_k=top_k_clauses)
    hard_critic = HardComplianceCritiqueAgent()
    soft_critic = SoftRiskCritiqueAgent()
    escalator = EscalationAgent(risk_threshold=risk_threshold)
    audit_builder = AuditChainBuilderAgent()

    audit_chains: List[Dict[str, Any]] = []

    for scenario in scenarios:
        combined_text = scenario.user_message + " " + scenario.assistant_response
        retrieved_clauses = retriever(combined_text)
        hard_result = hard_critic(scenario, retrieved_clauses)
        soft_result = soft_critic(scenario)
        escalation_result = escalator(scenario, hard_result, soft_result)
        audit_chain: AuditChain = audit_builder(
            scenario,
            retrieved_clauses,
            hard_result,
            soft_result,
            escalation_result,
        )
        audit_chains.append(audit_chain_to_dict(audit_chain))

    save_json(audit_chains, output_path)
