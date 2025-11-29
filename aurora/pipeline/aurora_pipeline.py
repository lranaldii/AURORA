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

    # --------------------- Shared LLM ---------------------
    llm = OpenAILLM(model=model_name)

    # --------------------- Initialise Agents ---------------------
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

    # ------------------------ main-loop -------------------------
    for scenario in scenarios:

        # 1. RETRIEVER 
        retrieval_output = retriever(scenario)
        retrieved_clauses: List[Clause] = retrieval_output["clauses"]
        retrieval_meta: Dict[str, Any] = retrieval_output  # already a dict with needed keys

        # 2. HARD COMPLIANCE 
        hard_result = hard_critic(scenario, retrieved_clauses)

        # 3. SOFT RISK ANALYSIS  
        soft_result = soft_critic(
            scenario,
            retrieval_meta=retrieval_meta,
            hard_result=hard_result,
        )

        # 4. ESCALATION DECISION
        escalation_result = escalator(
            scenario,
            hard_result,
            soft_result,
            retrieval_meta,
        )

        # 5. BUILD AUDIT CHAIN
        audit_chain: AuditChain = audit_builder(
            scenario,
            retrieved_clauses,
            hard_result,
            soft_result,
            escalation_result,
            retrieval_meta,
        )

        audit_chains.append(audit_chain_to_dict(audit_chain))

    # 6. SAVE OUTPUT
    save_json(audit_chains, output_path)
