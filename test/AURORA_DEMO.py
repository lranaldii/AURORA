# AURORA: Demonstration Notebook
# -----------------------------------------------------
# This notebook illustrates how AURORA can be used 
# for clause-grounded regulatory assurance on 
# financial LLM-based systems.
# 
# The demonstration follows the architecture described 
# in the AURORA paper (Figure 1), showing each step:
#   - Regulatory KB loading
#   - Hybrid clause retrieval
#   - Hard & soft compliance critics
#   - Escalation modelling
#   - Audit-chain generation
#   - Iterative refinement
#
# The example scenario illustrates a high-risk case.
# -----------------------------------------------------

import json

from aurora.utils.data_models import load_kb_from_json, load_scenarios_from_jsonl
from aurora.pipeline.aurora_pipeline import run_aurora_pipeline
from aurora.pipeline.iterative_pipeline import IterativeAuroraPipeline

from aurora.llm.openai_llm import OpenAILLM
from aurora.agents.clause_retrieval import HybridRAGClauseRetrievalAgent
from aurora.agents.hard_compliance_critic import HardComplianceCritiqueAgent
from aurora.agents.soft_risk_critic import SoftRiskCritiqueAgent
from aurora.agents.audit_agent import EscalationAgent
from aurora.agents.audit_chain_builder import AuditChainBuilderAgent

KB_PATH = "aurora/data/regulatory_kb.json"
SCENARIOS_PATH = "aurora/data/aurora_bench.jsonl"

kb = load_kb_from_json(KB_PATH)
scenarios = load_scenarios_from_jsonl(SCENARIOS_PATH)

print(f"Loaded {len(kb)} clauses from KB")
print(f"Loaded {len(scenarios)} scenarios")


# For demonstration, we use OpenAI; 
# can be replaced with HuggingFaceLLM
llm = OpenAILLM(
    model="gpt-4o-mini",
    temperature=0.2,
)


example = scenarios[0]

retriever = HybridRAGClauseRetrievalAgent(kb, top_k=5, threshold=0.35)
hard = HardComplianceCritiqueAgent()
soft = SoftRiskCritiqueAgent(llm)
escalator = EscalationAgent(risk_threshold=0.5)
builder = AuditChainBuilderAgent(llm)

combined_text = f"{example.user_message} {example.assistant_response}"

retrieval_out = retriever(combined_text)
retrieved_clauses = retrieval_out["clauses"]

retrieval_meta = {
    "retrieval_confidence": retrieval_out.get("retrieval_confidence"),
    "used_web_fallback": retrieval_out.get("used_web_fallback", False),
    "retrieval_failed": retrieval_out.get("retrieval_failed", False),
}

hard_out = hard(example, retrieved_clauses, retrieval_meta)
soft_out = soft(example, retrieval_meta, hard_out)
escalate = escalator(example, hard_out, soft_out, retrieval_meta)

audit_chain = builder(
    example,
    retrieved_clauses,
    hard_out,
    soft_out,
    escalate,
    retrieval_meta,
)

# Inspect the audit chain
audit_chain


from pprint import pprint

pprint(audit_chain.to_dict())


print(json.dumps(audit_chain.to_dict(), indent=2, ensure_ascii=False))


run_aurora_pipeline(
    KB_PATH,
    SCENARIOS_PATH,
    output_path="aurora_demo_output.json",
    top_k_clauses=5,
    retrieval_threshold=0.35
)

print("Saved audit chains to aurora_demo_output.json")


pipeline = IterativeAuroraPipeline(
    kb=kb,
    llm=llm,
    top_k_clauses=5,
    retrieval_threshold=0.35,
    risk_threshold=0.5,
    max_iterations=2
)

iterative_trail = pipeline.run(example)

print(f"Produced {len(iterative_trail)} refinement iterations")


for idx, chain in enumerate(iterative_trail):
    print("="*80)
    print(f"ITERATION {idx}")
    print("="*80)
    pprint(chain.to_dict())
    print()




final_response = iterative_trail[-1].improvement_suggestion.get(
    "replacement_guidance",
    ""
)

print("Final refined assistant response:\n")
print(final_response)


with open("aurora_demo_iterative_output.json", "w", encoding="utf-8") as f:
    json.dump([chain.to_dict() for chain in iterative_trail], f, ensure_ascii=False, indent=2)

print("Saved iterative audit trail to aurora_demo_iterative_output.json")


