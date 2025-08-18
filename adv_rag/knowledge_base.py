import uuid
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from adv_rag.utils import _slugify, chunk_text
from adv_rag.file_parser import FileParser
from adv_rag.web_scraper import WebScraper
from adv_rag.models import get_embedder
from adv_rag.config import RAGConfig


class KnowledgeBase:
    def __init__(self, files=None, dirs=None, links=None, name="default", cfg=None):
        self.cfg = cfg or RAGConfig()
        self.name = _slugify(name)

        self.files = [Path(p) for p in (files or []) if Path(p).is_file()]
        self.dirs = [Path(d) for d in (dirs or []) if Path(d).exists()]
        self.links = links or []
        self.scraper = WebScraper()

        # Embedding model
        self.embedding = get_embedder(self.cfg.embedding_model_name)
        self.dim = self.embedding.get_sentence_embedding_dimension()

        # FAISS index (cosine similarity via inner product)
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = []   # store ids to map back to docs
        self.doc_store = {}  # id -> (content, meta)

        # BM25 (optional lexical fallback)
        self.bm25 = None
        self.bm25_meta = []

        # Ingest immediately if files/dirs/links given
        if self.files or self.dirs or self.links:
            self.ingest()

    def ingest(self):
        docs = []

        # Collect docs from dirs
        for d in self.dirs:
            for p in d.rglob("*"):
                if p.is_file():
                    ct = FileParser.read(p)
                    if ct:
                        docs.append(
                            {"content": ct,
                             "meta": {"title": p.name, "source": str(p), "type": "file"}}
                        )

        # Collect docs from files
        for p in self.files:
            ct = FileParser.read(p)
            if ct:
                docs.append(
                    {"content": ct,
                     "meta": {"title": p.name, "source": str(p), "type": "file"}}
                )

        # Scrape links
        for u in self.links:
            t = self.scraper.fetch(u)
            if t:
                docs.append(
                    {"content": t,
                     "meta": {"title": u, "source": u, "type": "web"}}
                )

        if not docs:
            print("[INGEST] nothing to ingest")
            return

        # Chunk and embed
        chunks, metas, ids = [], [], []
        for d in docs:
            for i, ch in enumerate(chunk_text(
                d["content"],
                self.cfg.chunk_size_tokens,
                self.cfg.chunk_overlap_tokens,
            )):
                chunks.append(ch)
                ids.append(str(uuid.uuid4()))
                metas.append({**d["meta"], "chunk_id": i})

        embs = self.embedding.encode(chunks, batch_size=64, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")

        # Add to FAISS
        self.index.add(embs)
        self.id_map.extend(ids)
        for i, (doc, meta) in enumerate(zip(chunks, metas)):
            self.doc_store[ids[i]] = (doc, meta)

        # Setup BM25 if enabled
        if self.cfg.bm25_enable and BM25Okapi:
            corpus = [c.split() for c in chunks]
            self.bm25 = BM25Okapi(corpus)
            self.bm25_meta = list(zip(chunks, metas))

    @lru_cache(maxsize=256)
    def _embed_query(self, q: str):
        return np.array(self.embedding.encode([q], normalize_embeddings=True), dtype="float32")

    def hybrid(self, q: str, vec_k: int = 10, pool_k: int = 10):
        q_emb = self._embed_query(q)

        # FAISS dense retrieval
        sims, idxs = self.index.search(q_emb, vec_k)
        dense_hits = []
        for score, idx in zip(sims[0], idxs[0]):
            if idx == -1:  # FAISS padding
                continue
            doc_id = self.id_map[idx]
            doc, meta = self.doc_store[doc_id]
            dense_hits.append({"content": doc, "meta": meta, "score": float(score)})

        # BM25 lexical retrieval
        sparse_hits = []
        if self.bm25:
            scores = self.bm25.get_scores(q.split())
            top_idx = np.argsort(scores)[::-1][:vec_k]
            sparse_hits = [
                {"content": self.bm25_meta[i][0], "meta": self.bm25_meta[i][1], "score": float(scores[i])}
                for i in top_idx if scores[i] > 0
            ]

        # Merge with deduplication
        out, seen = [], set()
        for it in dense_hits + sparse_hits:
            key = (it["meta"].get("source"), it["meta"].get("chunk_id"))
            if key not in seen:
                seen.add(key)
                out.append(it)
                if len(out) >= pool_k:
                    break
        return out
