from typing import List, Dict, Any
import json

from aurora.utils.data_models import Scenario, Clause, AuditChain
from aurora.llm.base import BaseLLM


AUDIT_PROMPT = """
You are AURORA, a regulatory assurance assistant for financial services.

Given:
- a user message,
- an assistant response,
- a list of relevant regulatory clauses, and
- the outputs of hard and soft compliance critics,

you must produce a structured AUDIT CHAIN that documents:
1. Key factual elements of the interaction.
2. Detected risks or breaches.
3. How the interaction relates to the retrieved clauses.
4. A concise compliance assessment.
5. Concrete actions for developers or supervisors.
6. A safer, clause-aligned replacement response.

Respond ONLY with JSON in the following format:

{
  "extracted_facts": ["...", "..."],
  "detected_risks": ["...", "..."],
  "compliance_assessment": {
    "label": "COMPLIANT" | "BREACH" | "HIGH_RISK" | "MISSING_OBLIGATION",
    "rationale": "short justification"
  },
  "required_actions": ["...", "..."],
  "improvement_suggestion": {
    "replacement_guidance": "rewritten, compliant response to the user",
    "style_notes": "short notes on tone and disclosure"
  }
}

User message:
{user}

Assistant response:
{assistant}

Retrieved clauses (JSON list):
{clauses_json}

Hard critic output (JSON):
{hard_json}

Soft critic output (JSON):
{soft_json}

Retrieval meta-data (JSON):
{retrieval_json}
"""


class AuditChainBuilderAgent:
    """
    Uses an LLM to construct an AuditChain object from scenario + agent outputs.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def __call__(
        self,
        scenario: Scenario,
        retrieved_clauses: List[Clause],
        hard_result: Dict[str, Any],
        soft_result: Dict[str, Any],
        escalation_result: Dict[str, Any],
        retrieval_meta: Dict[str, Any] | None = None,
    ) -> AuditChain:

        clauses_payload = [
            {
                "clause_id": c.clause_id,
                "short_name": c.short_name,
                "obligation_type": c.obligation_type,
                "summary": c.summary,
            }
            for c in retrieved_clauses
        ]

        hard_json = json.dumps(hard_result, ensure_ascii=False)
        soft_json = json.dumps(soft_result, ensure_ascii=False)
        clauses_json = json.dumps(clauses_payload, ensure_ascii=False)
        retrieval_json = json.dumps(retrieval_meta or {}, ensure_ascii=False)

        prompt = AUDIT_PROMPT.format(
            user=scenario.user_message,
            assistant=scenario.assistant_response,
            clauses_json=clauses_json,
            hard_json=hard_json,
            soft_json=soft_json,
            retrieval_json=retrieval_json,
        )

        # Default fallback values
        default_label = scenario.compliance_label or "COMPLIANT"
        default_assessment = {
            "label": default_label,
            "rationale": scenario.notes
            or "Assessment derived from annotation and heuristics.",
        }
        default_actions = []

        if escalation_result.get("escalate", False):
            default_actions.append("Escalate this interaction to a human reviewer.")

        if hard_result.get("is_non_compliant", False):
            default_actions.append(
                "Review prompt templates and safety settings for this type of scenario."
            )

        default_improvement = {
            "replacement_guidance": (
                "Provide balanced, compliant guidance; avoid guarantees; "
                "highlight key risks; encourage regulated advice when appropriate."
            ),
            "style_notes": "Use clear, non-technical language; explicitly mention uncertainty.",
        }

        extracted_facts = [
            scenario.user_message.strip(),
            scenario.assistant_response.strip(),
        ]
        detected_risks = []

        # ------------------------ CALL LLM ------------------------
        try:
            raw = self.llm.generate([
                {"role": "user", "content": prompt}
            ])
            llm_out = json.loads(raw)
        except Exception:
            llm_out = {}

        extracted_facts = llm_out.get("extracted_facts", extracted_facts)
        detected_risks = llm_out.get("detected_risks", detected_risks)
        compliance_assessment = llm_out.get("compliance_assessment", default_assessment)
        required_actions = llm_out.get("required_actions", default_actions)
        improvement_suggestion = llm_out.get(
            "improvement_suggestion", default_improvement
        )

        return AuditChain(
            scenario_id=scenario.scenario_id,
            extracted_facts=extracted_facts,
            detected_risks=detected_risks,
            linked_clauses=clauses_payload,
            compliance_assessment=compliance_assessment,
            required_actions=required_actions,
            escalation_decision=escalation_result,
            improvement_suggestion=improvement_suggestion,
        )
