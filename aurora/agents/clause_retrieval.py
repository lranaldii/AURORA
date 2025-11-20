"""
Clause Retrieval Agent
This version implements a hybrid RAG mechanism:
1. Local embedding-based retrieval (free, using SentenceTransformers)
2. Web-search fallback retrieval using DuckDuckGo
3. Re-embedding of web results and final top-k ranking
"""

from typing import List, Dict, Any
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util

from aurora.utils.data_models import Clause


# -------------------------------------------------------
# 1. WEB SEARCH AGENT
# -------------------------------------------------------

class DuckDuckGoSearchAgent:
    """
    Free web search agent using DuckDuckGo's unofficial JSON endpoint.
    Acts as a fallback when the internal KB gives insufficient similarity.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def search(self, query: str) -> List[str]:
        """
        Returns: list of text snippets from DDG results
        """
        url = f"https://duckduckgo.com/?q={query}&format=json&pretty=1"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()

            snippets = []
            for entry in data.get("RelatedTopics", []):
                if isinstance(entry, dict) and "Text" in entry:
                    snippets.append(entry["Text"])
                if len(snippets) >= self.top_k:
                    break
            return snippets

        except Exception:
            return []


# -------------------------------------------------------
# 2. HYBRID RAG CLAUSE RETRIEVAL AGENT
# -------------------------------------------------------

class HybridRAGClauseRetrievalAgent:
    """
    Hybrid RAG system for clause retrieval:

    Step 1: Compute similarity between query and internal KB using embeddings
    Step 2: If the best score < threshold, trigger web search fallback
    Step 3: Re-embed web results + KB + mix them
    Step 4: Return top-K most relevant clauses from KB

    This produces clause retrieval that:
    - is free
    - is agentic
    - uses both local KB and external knowledge sources
    """

    def __init__(
        self,
        kb: List[Clause],
        top_k: int = 5,
        threshold: float = 0.35,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        web_top_k: int = 5
    ):
        self.kb = kb
        self.top_k = top_k
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings for KB
        self.kb_texts = [f"{c.short_name}: {c.summary}" for c in kb]
        self.kb_emb = self.model.encode(self.kb_texts, convert_to_tensor=True)

        # Web agent
        self.web_agent = DuckDuckGoSearchAgent(top_k=web_top_k)

    # -------------------------------------------------------

    def retrieve_from_kb(self, query_emb):
        """
        Returns: (scores, indices)
        """
        scores = util.cos_sim(query_emb, self.kb_emb)[0]
        np_scores = scores.cpu().numpy()
        top_idx = np.argsort(np_scores)[::-1]
        return np_scores, top_idx

    # -------------------------------------------------------

    def __call__(self, text: str) -> List[Clause]:
        """
        Hybrid retrieval pipeline:
        1. Internal embedding-based retrieval
        2. If confidence < threshold → web-search fallback
        3. Re-embed web snippets and mix with KB results
        """

        # 1. Encode query
        query_emb = self.model.encode(text, convert_to_tensor=True)

        # 2. Retrieve from KB
        kb_scores, kb_rank = self.retrieve_from_kb(query_emb)
        best_kb_score = float(kb_scores[kb_rank[0]])

        # If KB retrieval is strong enough → return KB-only result
        if best_kb_score >= self.threshold:
            return [self.kb[i] for i in kb_rank[: self.top_k]]

        # Otherwise: FALLBACK PATH
        query = text.replace("\n", " ")

        # Web search agent
        snippets = self.web_agent.search(query)

        if not snippets:
            # Web failed → return KB anyway
            return [self.kb[i] for i in kb_rank[: self.top_k]]

        web_emb = self.model.encode(snippets, convert_to_tensor=True)
        web_scores = util.cos_sim(query_emb, web_emb)[0].cpu().numpy()

        # Combine KB + web → soft RAG
        kb_scores_norm = (kb_scores - kb_scores.min()) / (kb_scores.max() - kb_scores.min() + 1e-9)
        web_scores_norm = (web_scores - web_scores.min()) / (web_scores.max() - web_scores.min() + 1e-9)

        combined_scores = kb_scores_norm + kb_scores_norm.mean() * web_scores_norm.mean()

        final_rank = np.argsort(combined_scores)[::-1]  # descending
        return [self.kb[i] for i in final_rank[: self.top_k]]
