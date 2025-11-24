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

from aurora.agents.clause_retrieval import HybridRAGClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.escalation_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent
from aurora.llm.openai_llm import OpenAILLM


def run_aurora_pipeline(
    kb_path: str,
    scenarios_path: str,
    output_path: str,
    top_k_clauses: int = 5,
    retrieval_threshold: float = 0.35,
    risk_threshold: float = 0.5,
    model_name: str = "gpt-4o-mini",
) -> None:

    kb: List[Clause] = load_kb_from_json(kb_path)
    scenarios: List[Scenario] = load_scenarios_from_jsonl(scenarios_path)

    # --------------------- LLM shared across all agents ---------------------
    llm = OpenAILLM(model=model_name)

    # ----------------------- Initialize agents ------------------------------
    retriever = HybridRAGClauseRetrievalAgent(
        kb,
        top_k=top_k_clauses,
        threshold=retrieval_threshold,
    )
    hard_critic = HardComplianceCritiqueAgent()
    soft_critic = SoftRiskCritiqueAgent(llm=llm)
    escalator = EscalationAgent(risk_threshold=risk_threshold)
    audit_builder = AuditChainBuilderAgent(llm=llm)

    audit_chains: List[Dict[str, Any]] = []

    for scenario in scenarios:
        combined_text = f"{scenario.user_message} {scenario.assistant_response}"

        retrieval_output = retriever(combined_text)
        retrieved_clauses: List[Clause] = retrieval_output["clauses"]
        retrieval_meta: Dict[str, Any] = {
            "retrieval_confidence": retrieval_output.get("retrieval_confidence"),
            "used_web_fallback": retrieval_output.get("used_web_fallback", False),
            "retrieval_failed": retrieval_output.get("retrieval_failed", False),
        }

        hard_result = hard_critic(scenario, retrieved_clauses, retrieval_meta)
        soft_result = soft_critic(scenario, retrieval_meta, hard_result)
        escalation_result = escalator(
            scenario,
            hard_result,
            soft_result,
            retrieval_meta,
        )

        audit_chain: AuditChain = audit_builder(
            scenario,
            retrieved_clauses,
            hard_result,
            soft_result,
            escalation_result,
            retrieval_meta,
        )

        audit_chains.append(audit_chain_to_dict(audit_chain))

    save_json(audit_chains, output_path)
