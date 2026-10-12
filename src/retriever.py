"""
retriever.py
-------------
Semantic retrieval over the corpus using sentence-transformers.

Step 1 — Index: embed all clause texts using a local embedding model.
Step 2 — Retrieve: given a query string, return top-k most similar clauses.

We use 'all-MiniLM-L6-v2' (lightweight, fast, no API key required).
Embeddings are cached to disk (index/embeddings.npy + index/clause_ids.json)
so re-runs don't re-embed the full corpus.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.corpus_loader import Clause


# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"   # 22M params, 384-dim embeddings
INDEX_DIR  = Path(__file__).parent.parent / "index"   # project-root/index/


# ── Retriever Class ──────────────────────────────────────────────────────────

class Retriever:
    """
    Wraps a sentence-transformer model for semantic clause retrieval.

    Usage:
        retriever = Retriever(clauses)
        results   = retriever.search("What is the WFH policy?", k=10)
    """

    def __init__(self, clauses: list[Clause], rebuild_index: bool = False):
        """
        Args:
            clauses:       All Clause objects from corpus_loader.
            rebuild_index: Force re-embedding even if cache exists.
        """
        self.clauses   = clauses
        self.clause_map = {c.clause_id: c for c in clauses}  # fast lookup by clause_id
        self.model     = SentenceTransformer(MODEL_NAME)

        # Load or build the embedding index
        self._embeddings = self._load_or_build(rebuild_index)

    # ── Index Building ────────────────────────────────────────────────────

    def _load_or_build(self, rebuild: bool) -> np.ndarray:
        """Load cached embeddings or embed the corpus from scratch."""
        INDEX_DIR.mkdir(exist_ok=True)
        emb_path = INDEX_DIR / "embeddings.npy"
        ids_path = INDEX_DIR / "clause_ids.json"

        if not rebuild and emb_path.exists() and ids_path.exists():
            # Validate cached index matches current corpus
            cached_ids = json.loads(ids_path.read_text())
            current_ids = [c.clause_id for c in self.clauses]
            if cached_ids == current_ids:
                print("[Retriever] Loading cached embeddings from disk...")
                return np.load(str(emb_path))

        # Build fresh index
        print(f"[Retriever] Embedding {len(self.clauses)} clauses with '{MODEL_NAME}'...")
        texts = [c.text for c in self.clauses]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine sim becomes dot product
        )
        # Cache to disk
        np.save(str(emb_path), embeddings)
        ids_path.write_text(json.dumps([c.clause_id for c in self.clauses]))
        print(f"[Retriever] Index built and cached -> {INDEX_DIR}")
        return embeddings

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[tuple[float, Clause]]:
        """
        Return the top-k clauses most semantically similar to the query.

        Returns:
            List of (score, Clause) sorted descending by cosine similarity.
        """
        # Embed the query (normalised so dot product = cosine similarity)
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        # Compute cosine similarities via dot product (embeddings already normalised)
        scores = self._embeddings @ q_emb   # shape (N,)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        return [(float(scores[i]), self.clauses[i]) for i in top_k_idx]
