from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json


@dataclass
class Clause:
    clause_id: str
    regime: str
    short_name: str
    obligation_type: str
    summary: str
    keywords: List[str]
    jurisdiction: str
    risk_level: str


@dataclass
class Scenario:
    scenario_id: str
    user_message: str
    assistant_response: str
    compliance_label: str
    notes: str = ""
    escalation_required: bool = False
    linked_clauses: List[str] = None
    task_type: str = "dialogue"  # "definition", "reg_qa", "xbrl", "cdm", "mof"
    gold_answer: str | None = None  # canonic answer
    metadata: Dict[str, Any] = None  # tag XBRL, id CDM, MOF



@dataclass
class AuditChain:
    scenario_id: str
    extracted_facts: List[str]
    detected_risks: List[str]
    linked_clauses: List[Dict[str, Any]]
    compliance_assessment: Dict[str, Any]
    required_actions: List[str]
    escalation_decision: Dict[str, Any]
    improvement_suggestion: Dict[str, Any]


def simple_tokenise(text: str) -> List[str]:
    return [t.lower().strip(".,?!:;\"'()[]") for t in text.split() if t.strip()]


def load_kb_from_json(path: str) -> List[Clause]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Clause(**c) for c in data]


def load_scenarios_from_jsonl(path: str) -> List[Scenario]:
    scenarios: List[Scenario] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scenarios.append(Scenario(**d))
    return scenarios


def audit_chain_to_dict(a: AuditChain) -> Dict[str, Any]:
    return asdict(a)
