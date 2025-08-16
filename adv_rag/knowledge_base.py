import uuid
from pathlib import Path
from functools import lru_cache

import chromadb

try:
    from rank_bm25 import BM25Okapi
except:
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

        self.files = [Path(p) for p in (files or [])]
        self.dirs = [Path(d) for d in (dirs or [])]
        self.links = links or []
        self.scraper = WebScraper()

        # DB path per agent
        self.db_path = Path(self.cfg.chroma_path) / self.name
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Collection name per agent
        self.collection_name = f"{self.cfg.collection_name}_{self.name}"

        self.chroma = chromadb.PersistentClient(path=str(self.db_path))

        metadata = {"hnsw:space": "cosine"}
        try:
            self.coll = self.chroma.get_collection(self.collection_name)
        except Exception:
            self.coll = self.chroma.create_collection(
                name=self.collection_name, metadata=metadata
            )

        self.embedding = get_embedder(self.cfg.embedding_model_name)

        self.bm25 = None
        self.bm25_meta = []

        if self.coll.count() == 0:
            self.ingest()
        else:
            self._rebuild_bm25_from_store()

    def ingest(self):
        docs = []

        # dirs
        for d in self.dirs:
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file():
                        ct = FileParser.read(p)
                        if ct:
                            docs.append(
                                {"content": ct,
                                 "meta": {"title": p.name, "source": str(p), "type": "file"}}
                            )

        # files
        for p in self.files:
            if p.is_file():
                ct = FileParser.read(p)
                if ct:
                    docs.append(
                        {"content": ct,
                         "meta": {"title": p.name, "source": str(p), "type": "file"}}
                    )

        # links
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

        chunks, metas, ids = [], [], []
        for d in docs:
            for i, ch in enumerate(chunk_text(d["content"], self.cfg.chunk_size_tokens, self.cfg.chunk_overlap_tokens)):
                chunks.append(ch)
                m = dict(d["meta"])
                m["chunk_id"] = i
                metas.append(m)
                ids.append(str(uuid.uuid4()))

        embs = self.embedding.encode(
            chunks, batch_size=64, normalize_embeddings=True
        ).tolist()

        self.coll.add(
            documents=chunks,
            embeddings=embs,
            metadatas=metas,
            ids=ids,
        )

        if self.cfg.bm25_enable and BM25Okapi:
            corpus = [c.split() for c in chunks]
            self.bm25 = BM25Okapi(corpus)
            self.bm25_meta = list(zip(chunks, metas))

    def _rebuild_bm25_from_store(self):
        if not (self.cfg.bm25_enable and BM25Okapi):
            return
        res = self.coll.get(include=["documents", "metadatas"])
        docs = res.get("documents", []) or []
        mets = res.get("metadatas", []) or []
        if not docs:
            return
        corpus = [d.split() for d in docs]
        self.bm25 = BM25Okapi(corpus)
        self.bm25_meta = list(zip(docs, mets))

    @lru_cache(maxsize=256)
    def _embed_query(self, q: str):
        return self.embedding.encode([q], normalize_embeddings=True).tolist()

    def hybrid(self, q: str, vec_k: int, pool_k: int):
        vec_emb = self._embed_query(q)
        res = self.coll.query(query_embeddings=vec_emb, n_results=vec_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        primary = [
            {"content": d, "meta": m, "score": 1.0 - dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

        bm = []
        if self.bm25:
            toks = q.split()
            sc = self.bm25.get_scores(toks)
            idx = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:vec_k]
            bm = [
                {"content": self.bm25_meta[i][0], "meta": self.bm25_meta[i][1], "score": float(sc[i])}
                for i in idx
            ]

        out, seen = [], set()
        for it in primary + bm:
            key = (it["meta"].get("source"), it["meta"].get("chunk_id"))
            if key not in seen:
                seen.add(key)
                out.append(it)
                if len(out) >= pool_k:
                    break
        return out
