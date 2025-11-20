import json
import argparse
from aurora.utils.data_models import load_kb_from_json, load_scenarios_from_jsonl


def validate_kb(path: str) -> None:
    kb = load_kb_from_json(path)
    print(f"Loaded {len(kb)} clauses from {path}.")
    clause_ids = {c.clause_id for c in kb}
    if len(clause_ids) != len(kb):
        print("WARNING: duplicate clause_id entries detected.")
    else:
        print("No duplicate clause_ids detected.")


def validate_scenarios(path: str) -> None:
    scenarios = load_scenarios_from_jsonl(path)
    print(f"Loaded {len(scenarios)} scenarios from {path}.")
    scenario_ids = {s.scenario_id for s in scenarios}
    if len(scenario_ids) != len(scenarios):
        print("WARNING: duplicate scenario_id entries detected.")
    else:
        print("No duplicate scenario_ids detected.")
    # Basic schema checks
    for s in scenarios:
        if s.compliance_label not in {
            "COMPLIANT",
            "BREACH",
            "HIGH_RISK",
            "MISSING_OBLIGATION",
        }:
            print(f"WARNING: unexpected compliance_label '{s.compliance_label}' in {s.scenario_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate AURORA data files.")
    parser.add_argument(
        "--kb",
        type=str,
        default="aurora/data/regulatory_kb.json",
        help="Path to regulatory KB JSON.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="aurora/data/aurora_scenarios.jsonl",
        help="Path to scenarios JSONL.",
    )
    args = parser.parse_args()

    validate_kb(args.kb)
    validate_scenarios(args.scenarios)
