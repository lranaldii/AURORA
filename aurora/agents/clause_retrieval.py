"""
Clause Retrieval Agents for AURORA.

This module implements a hybrid RAG mechanism:
- Embedding-based retrieval over the internal regulatory KB.
- Optional web-search fallback using DuckDuckGo.
- Query expansion with web snippets and re-ranking.
- A retrieval meta-signal capturing confidence and fallback status.

The HybridRAGClauseRetrievalAgent returns a dictionary with:
- "clauses": List[Clause]
- "retrieval_confidence": float in [0, 1] (approximate)
- "used_web_fallback": bool
- "retrieval_failed": bool
"""

from typing import List, Dict, Any
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util

from aurora.utils.data_models import Clause


class HybridRAGClauseRetrievalAgent:
    """
    Hybrid RAG system for clause retrieval.

    Step 1: Compute similarity between query and internal KB using embeddings.
    Step 2: If the best score < threshold and web fallback is enabled, issue a web search and extract a short snippet.
    Step 3: Expand the query with the snippet and recompute similarities.
    Step 4: Return top-K most relevant clauses from the KB, together with a retrieval meta-signal describing confidence and fallback use.
    """

    def __init__(
        self,
        kb: List[Clause],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        threshold: float = 0.35,
        use_web_fallback: bool = True,
        duckduckgo_url: str = "https://duckduckgo.com/html/",
    ) -> None:
        self.kb = kb
        self.top_k = top_k
        self.threshold = threshold
        self.use_web_fallback = use_web_fallback
        self.duckduckgo_url = duckduckgo_url

        # Pre-compute embeddings for all clauses in the KB.
        self.model = SentenceTransformer(model_name)
        self._kb_texts = [f"{c.short_name}. {c.summary}" for c in kb]
        if self._kb_texts:
            self._kb_embeddings = self.model.encode(
                self._kb_texts, convert_to_tensor=True
            )
        else:
            self._kb_embeddings = None

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    def _strip_html(self, html: str) -> str:
        """
        Very lightweight HTML stripper to avoid an additional dependency.
        It removes tags and collapses whitespace.
        """
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _web_search_snippet(self, query: str, max_chars: int = 2000) -> str:
        """
        Query DuckDuckGo and return a single text snippet suitable for query expansion. 
        This is intentionally simple: the goal is to enrich the query with additional regulatory language.
        """
        try:
            params = {"q": query, "kl": "uk-en"}
            resp = requests.get(self.duckduckgo_url, params=params, timeout=10)
            if resp.status_code != 200:
                return ""
            clean = self._strip_html(resp.text)
            if len(clean) > max_chars:
                clean = clean[:max_chars]
            return clean
        except Exception:
            # In production we would log the error; here we fail silently.
            return ""

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> Dict[str, Any]:
        """
        Run hybrid retrieval for the given free-text query.

        Returns a dictionary with:
        - "clauses": List[Clause]
        - "retrieval_confidence": float
        - "used_web_fallback": bool
        - "retrieval_failed": bool
        """
        if not self.kb or self._kb_embeddings is None:
            return {
                "clauses": [],
                "retrieval_confidence": 0.0,
                "used_web_fallback": False,
                "retrieval_failed": True,
            }

        # Embed query and compute cosine similarity with KB.
        query_emb = self.model.encode(text, convert_to_tensor=True)
        kb_scores = util.cos_sim(query_emb, self._kb_embeddings)[0].cpu().numpy()

        # Normalise scores to [0, 1] for interpretability.
        kb_min = float(kb_scores.min())
        kb_max = float(kb_scores.max())
        if kb_max - kb_min > 1e-9:
            kb_scores_norm = (kb_scores - kb_min) / (kb_max - kb_min)
        else:
            kb_scores_norm = np.zeros_like(kb_scores)

        best_idx = int(np.argmax(kb_scores_norm))
        best_score = float(kb_scores_norm[best_idx])

        # Initial ranking purely from KB.
        kb_rank = np.argsort(kb_scores_norm)[::-1]
        top_k_indices = kb_rank[: self.top_k]
        top_k_clauses = [self.kb[i] for i in top_k_indices]

        # Case 1: confident KB retrieval (no fallback)
        if best_score >= self.threshold or not self.use_web_fallback:
            return {
                "clauses": top_k_clauses,
                "retrieval_confidence": best_score,
                "used_web_fallback": False,
                "retrieval_failed": False,
            }

        # Case 2: low-confidence KB retrieval and attempt web fallback.
        snippet = self._web_search_snippet(
            text + " financial regulation FCA PRA Consumer Duty"
        )

        if not snippet:
            # Web fallback failed; we still return KB candidates but flag
            # the retrieval as potentially unreliable.
            return {
                "clauses": top_k_clauses,
                "retrieval_confidence": best_score,
                "used_web_fallback": False,
                "retrieval_failed": True,
            }

        # Expand the query with the web snippet and recompute similarities.
        expanded_query = f"{text}\n\nAdditional context:\n{snippet}"
        expanded_emb = self.model.encode(expanded_query, convert_to_tensor=True)
        kb_scores_expanded = util.cos_sim(expanded_emb, self._kb_embeddings)[0].cpu().numpy()

        kb_min2 = float(kb_scores_expanded.min())
        kb_max2 = float(kb_scores_expanded.max())
        if kb_max2 - kb_min2 > 1e-9:
            kb_scores_norm2 = (kb_scores_expanded - kb_min2) / (kb_max2 - kb_min2)
        else:
            kb_scores_norm2 = np.zeros_like(kb_scores_expanded)

        best_idx2 = int(np.argmax(kb_scores_norm2))
        best_score2 = float(kb_scores_norm2[best_idx2])

        final_rank = np.argsort(kb_scores_norm2)[::-1]
        final_indices = final_rank[: self.top_k]
        final_clauses = [self.kb[i] for i in final_indices]

        # Retrieval is considered failed if, even after expansion, all scores remain low. The threshold here is conservative.
        retrieval_failed = best_score2 < 0.10

        return {
            "clauses": final_clauses,
            "retrieval_confidence": max(best_score, best_score2),
            "used_web_fallback": True,
            "retrieval_failed": retrieval_failed,
        }
