from typing import List, Dict, Any
from adv_rag.models import get_reranker


class AdvancedRAG:
    def __init__(self, kb, llm=None, cfg=None):
        self.kb = kb
        self.cfg = cfg or kb.cfg
        self.rerank = get_reranker(self.cfg.reranker_model_name)
        self.llm = llm

    def retrieve(self, q: str, top_k: int = None):
        k = int(top_k or self.cfg.top_k_initial)
        pool = self.cfg.top_k_rerank_pool
        pool_items = self.kb.hybrid(q, vec_k=k * 3, pool_k=pool)
        if not pool_items:
            return []
        pairs = [(q, item["content"]) for item in pool_items]
        scores = self.rerank.predict(pairs)
        ranked = sorted(zip(pool_items, scores), key=lambda x: x[1], reverse=True)[:k]
        return [r for r, _ in ranked]

    def format(self, docs: List[Dict[str, Any]]) -> str:
        parts = []
        for d in docs:
            m = d["meta"]
            hd = f"[{m.get('type')}|{m.get('title')}|{m.get('source')}]"
            parts.append(f"{hd}\n{d['content']}")
        return "\n\n---\n\n".join(parts)
