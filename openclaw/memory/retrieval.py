"""
Hybrid Retrieval Engine
Combines semantic, keyword, and contextual search (MemU approach).
"""

import logging
import math
import re
import time
from collections import Counter
from typing import Optional

logger = logging.getLogger("openclaw.memory.retrieval")


class HybridRetrieval:
    """
    Multi-strategy retrieval combining:
    - Keyword matching (TF-IDF style)
    - Semantic similarity (when embeddings available)
    - Contextual relevance (recency, significance, access patterns)
    """

    def __init__(self, item_layer, category_layer):
        self.items = item_layer
        self.categories = category_layer

    async def search(self, query: str, top_k: int = 10, method: str = "hybrid") -> list[dict]:
        """
        Search memory using specified method.
        Methods: hybrid, keyword, semantic, contextual
        """
        if method == "keyword":
            return self._keyword_search(query, top_k)
        elif method == "contextual":
            return self._contextual_search(query, top_k)
        else:  # hybrid
            return self._hybrid_search(query, top_k)

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """Combine keyword and contextual scores."""
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2)

        # Get contextual results
        contextual_results = self._contextual_search(query, top_k * 2)

        # Get category results
        category_results = self.categories.search_categories(query, top_k)

        # Merge and score
        seen = set()
        merged = []

        for r in keyword_results:
            key = r.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                r["_final_score"] = r.get("_score", 0) * 0.5
                merged.append(r)

        for r in contextual_results:
            key = r.get("content", "")[:100]
            if key in seen:
                # Boost existing
                for m in merged:
                    if m.get("content", "")[:100] == key:
                        m["_final_score"] = m.get("_final_score", 0) + r.get("_score", 0) * 0.3
            else:
                seen.add(key)
                r["_final_score"] = r.get("_score", 0) * 0.3
                merged.append(r)

        for r in category_results:
            key = r.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                r["_final_score"] = 0.2
                merged.append(r)

        merged.sort(key=lambda x: x.get("_final_score", 0), reverse=True)

        # Clean up internal scores before returning
        for r in merged[:top_k]:
            r.pop("_score", None)
            r.pop("_final_score", None)

        return merged[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """TF-IDF inspired keyword search."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Get term frequencies across all items
        all_items = self.items.all_items()
        if not all_items:
            return []

        doc_count = len(all_items)
        term_doc_freq = Counter()
        for item in all_items:
            item_terms = set(self._tokenize(item.get("content", "")))
            for term in query_terms:
                if term in item_terms:
                    term_doc_freq[term] += 1

        # Score each item
        scored = []
        for item in all_items:
            item_terms = self._tokenize(item.get("content", ""))
            item_term_freq = Counter(item_terms)

            score = 0
            for term in query_terms:
                tf = item_term_freq.get(term, 0) / max(len(item_terms), 1)
                df = term_doc_freq.get(term, 0)
                idf = math.log((doc_count + 1) / (df + 1)) + 1
                score += tf * idf

            if score > 0:
                scored.append({**item, "_score": score})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:top_k]

    def _contextual_search(self, query: str, top_k: int) -> list[dict]:
        """Search based on significance, recency, and access patterns."""
        all_items = self.items.all_items()
        if not all_items:
            return []

        now = time.time()
        scored = []

        for item in all_items:
            significance = item.get("significance", 0.5)
            access_count = item.get("access_count", 0)
            age_hours = (now - item.get("created_at", now)) / 3600

            # Recency: exponential decay
            recency = math.exp(-age_hours / (720))  # ~30 day half-life

            # Access frequency
            freq_score = min(access_count / 10, 1.0)

            score = (significance * 0.5) + (recency * 0.3) + (freq_score * 0.2)
            scored.append({**item, "_score": score})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for keyword search."""
        # Remove punctuation and lowercase
        text = re.sub(r"[^\w\s]", " ", text.lower())
        # Split and filter stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "can", "could",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "and", "or", "not", "but", "if", "then",
            "le", "la", "les", "de", "du", "des", "un", "une", "et",
            "ou", "en", "dans", "sur", "avec", "par", "pour", "est",
            "sont", "a", "je", "tu", "il", "elle", "nous", "vous", "ils",
        }
        tokens = text.split()
        return [t for t in tokens if t not in stop_words and len(t) > 1]
