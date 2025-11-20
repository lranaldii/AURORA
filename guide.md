### *Benchmarking, running, and evaluating the AURORA oversight framework*

AURORA includes a comprehensive evaluation pipeline to assess the effectiveness of its multi-agent regulatory oversight components.
This document provides a step-by-step tutorial covering:

1. Preparing internal and external benchmark datasets
2. Running the AURORA multi-agent oversight pipeline
3. Evaluating clause retrieval, risk scoring, escalation quality, and audit-chain generation

This guide is intended for both researchers and practitioners who want to benchmark AURORA across compliance-specific and reasoning-focused financial datasets.

---

# 1. External Benchmarks

AURORA provides a utility script that converts several well-known financial NLP datasets into **AURORA-compatible oversight scenarios**.

The script:

```
scripts/prepare_external_benchmarks.py
```

automatically generates:

* **Banking77 scenarios** (intent-based, some high-risk)
* **FinQA scenarios** (numerical reasoning stress test)
* **ConvFinQA scenarios** (multi-turn financial QA stress test)
* **Manual pseudo-log scenarios** (mortgages, debt, fraud, vulnerability, get-rich-quick queries)

## Step 1 — Run the builder

```bash
python scripts/prepare_external_benchmarks.py \
  --output_dir data/external_benchmarks \
  --max_banking77 200 \
  --max_finqa 100 \
  --max_convfinqa 100
```

### This produces:

```
data/external_benchmarks/
├── banking77_scenarios.jsonl
├── finqa_scenarios.jsonl
├── convfinqa_scenarios.jsonl
└── manual_pseudolog_scenarios.jsonl
```

Each file contains a list of AURORA-style scenario items with:

* `scenario_id`
* `user_message`
* `assistant_response` (placeholder or gold)
* `compliance_label`
* `escalation_required`
* `linked_clauses` (empty by default)
* `notes`

These files are directly compatible with the AURORA pipeline.

---

# 2. Running AURORA Pipeline

You can execute the full multi-agent oversight pipeline using:

```
python scripts/run_pipeline.py
```

## Example: Run AURORA on Banking77

```bash
python scripts/run_pipeline.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/banking77_scenarios.jsonl \
  --output aurora/output/banking77_audit_chains.json
```

The pipeline:

1. Loads the Regulatory Knowledge Base
2. Loads the chosen scenario file
3. Runs each AURORA agent:

   * **HybridRAGClauseRetrievalAgent** (embeddings + web fallback)
   * **HardComplianceCritiqueAgent**
   * **SoftRiskCritiqueAgent (LLM-as-Judge)**
   * **EscalationAgent**
   * **AuditChainBuilderAgent (SORC structured reasoning)**
4. Generates a complete **clause-grounded audit chain** for each scenario
5. Saves all audit chains to a JSON file

Example output:

```
aurora/output/banking77_audit_chains.json
```

Each entry contains:

* extracted facts
* retrieved clauses
* violation analysis
* risk-level assignment
* escalation decision
* suggested corrected response
* meta-reflection

---

# 3. Evaluating AURORA

Evaluation is handled by:

```
scripts/run_evaluation.py
```

This script computes:

### Clause-retrieval scores

* Clause coverage
* Precision@3 / Precision@5
* Recall@3 / Recall@5
* (optional) MRR

### Escalation performance

* Accuracy
* Precision
* Recall
* F1

### Risk-scoring diagnostics (if using LLM-as-judge)

* Distribution of risk scores
* Correlation with gold escalation labels
* (optional) AUROC

## Example: Evaluate AURORA on Banking77

```bash
python scripts/run_evaluation.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/banking77_scenarios.jsonl \
  --audit_chains aurora/output/banking77_audit_chains.json
```

### Optional: Regenerate audit chains before evaluating

```bash
python scripts/run_evaluation.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/banking77_scenarios.jsonl \
  --audit_chains aurora/output/banking77_audit_chains.json \
  --run_pipeline
```

---

# 4. Testing with FinQA and ConvFinQA

FinQA and ConvFinQA are **not** regulatory datasets.
However, they are extremely valuable for:

* testing AURORA’s robustness on financial reasoning
* ensuring the multi-agent audit chain remains coherent for numerical scenarios
* verifying that clause retrieval behaves sensibly even when the content is non-regulatory
* measuring noise sensitivity in SORC-based oversight

## Run the pipeline on FinQA:

```bash
python scripts/run_pipeline.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/finqa_scenarios.jsonl \
  --output aurora/output/finqa_audit_chains.json
```

Then evaluate:

```bash
python scripts/run_evaluation.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/finqa_scenarios.jsonl \
  --audit_chains aurora/output/finqa_audit_chains.json
```

---

# 5. Manual Pseudo-Log Scenarios

AURORA includes hand-crafted templates simulating realistic high-risk interactions:

* job loss → mortgage affordability
* gambling addiction
* debt consolidation
* fraud attempts
* get-rich-quick investment schemes

These are ideal for:

* validating risk detection
* evaluating escalation
* testing audit chain quality
* performing ablation studies

To run:

```bash
python scripts/run_pipeline.py \
  --kb aurora/data/regulatory_kb.json \
  --scenarios data/external_benchmarks/manual_pseudolog_scenarios.jsonl \
  --output aurora/output/manual_audit_chains.json
```

---
