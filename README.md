# AURORA: A Multi-Agent Framework for AUditable Regulatory Oversight Reasoning in financial AI systems

AURORA is a modular, multi-agent system for real-time regulatory oversight in financial AI applications.
It generates clause-grounded audit chains, detects non-compliant model behaviour, retrieves regulatory obligations, evaluates risks, and performs transparent multi-agent reasoning.

## AURORA is designed for research on:

- law-following AI

- financial compliance

- governance & oversight

- transparent decision-making pipelines

- multi-agent systems with LLMs

- AI assurance in regulated environments

## Key Features

Multi-agent architecture coordinating specialised agents:

- Clause Retrieval

- Hard Compliance Critique

- Soft Risk Analysis

- Escalation Logic

- Audit Chain Construction

- Clause-grounded reasoning using a built-in Regulatory KB (FCA, PRA, EU AI Act, Internal rules)

- GPT-powered modules using the OpenAI API

- Annotated dataset (30+ realistic compliance scenarios)

- Modular codebase for easy extension

- Traceable audit chains to support human oversight

## Repo structure

```
aurora/
│
├── data/             # Regulatory KB + annotated scenarios
│── agents/           # Multi-agent system modules
│── models/           # LLM API/HF wrapper
│── pipeline/         # Full AURORA pipeline
│── evaluation/       # Metrics
│── utils/            # Helpers
└── scripts/          # Runner & tools
```
