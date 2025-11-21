from typing import Dict, Any
import openai

class LLMAnswerJudge:
    """
    LLM-as-Judge evaluator for semantic equivalence between a model answer and the gold answer.
    Returns a binary judgment: 1 (correct) or 0 (incorrect).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def __call__(self, predicted: str, gold: str) -> int:
        prompt = f"""
You are an expert evaluator for financial regulatory questions.

Determine whether the following two answers are semantically equivalent
for the purpose of scoring correctness. 
Return ONLY 1 (correct) or 0 (incorrect).

GOLD ANSWER:
{gold}

PREDICTED ANSWER:
{predicted}

Is the predicted answer fully consistent with the gold answer?
"""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )

        judgment = response["choices"][0]["message"]["content"].strip()

        # Ensure numeric clean output
        if judgment.startswith("1"):
            return 1
        else:
            return 0
