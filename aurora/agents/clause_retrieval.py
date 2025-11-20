from typing import List
from aurora.utils.data_models import Clause, simple_tokenise


class ClauseRetrievalAgent:
    """
    Simple keyword matching retriever over our Regulatory KB.
    Replaced or complemented with dense retrieval or RAG pipeline.
    """

    def __init__(self, kb: List[Clause], top_k: int = 5):
        self.kb = kb
        self.top_k = top_k

    def __call__(self, text: str) -> List[Clause]:
        tokens = set(simple_tokenise(text))
        scored: List[tuple[int, Clause]] = []
        for clause in self.kb:
            kw = set(k.lower() for k in clause.keywords)
            overlap = len(tokens & kw)
            if overlap > 0:
                scored.append((overlap, clause))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self.top_k]]
