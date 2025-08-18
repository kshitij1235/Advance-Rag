from typing import List, Dict, Any, Iterable, Tuple, Optional
from itertools import islice
import hashlib
import heapq
from collections import OrderedDict
from adv_rag.models import get_reranker

# Optional fast hash
try:
    import xxhash  # pip install xxhash
    _HAS_XXHASH = True
except Exception:
    _HAS_XXHASH = False

# Optional sane caching
try:
    from cachetools import TTLCache  # pip install cachetools
    _HAS_CACHETOOLS = True
except Exception:
    _HAS_CACHETOOLS = False


def batch_iterable(iterable: Iterable, batch_size: int) -> Iterable[List]:
    """
    Yield lists of size up to batch_size from any iterable.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def fast_hash(text: str) -> str:
    """
    Fast non-crypto-ish hash for dedup. Uses xxhash if available,
    otherwise blake2b with small digest (fast in stdlib).
    """
    if _HAS_XXHASH:
        return xxhash.xxh64(text).hexdigest()
    # blake2b is fast and in stdlib; 8 bytes digest is plenty for dedup buckets
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8)
    return h.hexdigest()


class _SimpleLRU:
    """
    Tiny LRU cache (per-instance) with a max size.
    Keyed by a tuple (query, top_k).
    """
    def __init__(self, maxsize: int = 256):
        self.maxsize = int(maxsize)
        self._data: "OrderedDict[Tuple[str, Optional[int]], List[Dict[str, Any]]]" = OrderedDict()

    def get(self, key):
        try:
            val = self._data.pop(key)
            # re-insert to mark as most-recent
            self._data[key] = val
            return val
        except KeyError:
            return None

    def set(self, key, value):
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self.maxsize:
            self._data.popitem(last=False)  # evict LRU
        self._data[key] = value


class AdvancedRAG:
    """
    Advanced Retrieval-Augmented Generation pipeline — without the bloat.

    - Hybrid retrieval via kb.hybrid(query, vec_k=..., pool_k=...) (your KB must provide this).
    - Dedup via cheap content hash (xxhash if installed).
    - Rerank batched via self.rerank.predict(pairs).
    - Top-k via heapq.nlargest (don’t sort the whole damned list).
    - Per-instance cache with TTL if cachetools is present, else a small LRU.
    - Clean formatting without dumb string concatenation loops.

    Expected doc structure from kb.hybrid():
      { "content": "text ...", "meta": {"type": "...", "title": "...", "source": "..."} }
    """

    def __init__(self, kb, llm=None, cfg=None):
        self.kb = kb
        self.cfg = cfg or kb.cfg
        # Your factory; assumed to return an object with .predict(List[Tuple[str,str]]) -> List[float]
        self.rerank = get_reranker(self.cfg.reranker_model_name)
        self.llm = llm

        # Cache setup
        cache_size = int(getattr(self.cfg, "retrieve_cache_size", 256))
        cache_ttl = int(getattr(self.cfg, "retrieve_cache_ttl", 0))  # seconds; 0 disables TTL
        if _HAS_CACHETOOLS and cache_ttl > 0:
            self._cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self._use_ttl = True
        else:
            self._cache = _SimpleLRU(maxsize=cache_size)
            self._use_ttl = False

    # --------------------------
    # Public API
    # --------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Cached wrapper around the core retrieval.
        """
        key = (query, int(top_k) if top_k is not None else None)
        if self._use_ttl:
            # cachetools cache acts like a dict
            cached = self._cache.get(key)  # type: ignore[attr-defined]
            if cached is not None:
                return cached
            result = self._retrieve_uncached(query, top_k)
            self._cache[key] = result  # type: ignore[index]
            return result
        else:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = self._retrieve_uncached(query, top_k)
            self._cache.set(key, result)
            return result

    def format(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for LLM input or display.
        """
        parts: List[str] = []
        for doc in docs:
            meta = doc.get("meta", {}) or {}
            doc_type = meta.get("type", "")
            doc_title = meta.get("title", "")
            doc_source = meta.get("source", "")
            header = f"[{doc_type}|{doc_title}|{doc_source}]"
            content = doc.get("content", "") or ""
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)

    # --------------------------
    # Internals
    # --------------------------
    def _retrieve_uncached(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Core retrieval pipeline without caching.
        """
        # Config
        k = int(top_k) if top_k is not None else int(self.cfg.top_k_initial)
        pool_size = int(getattr(self.cfg, "top_k_rerank_pool", k * 5))
        batch_size = int(getattr(self.cfg, "reranker_batch_size", 16))

        vec_k = min(k * 3, pool_size * 2)

        # 1) Hybrid retrieve
        try:
            pool_items: List[Dict[str, Any]] = self.kb.hybrid(
                query,
                vec_k=vec_k,
                pool_k=pool_size
            )
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return []

        if not pool_items:
            return []

        # 2) Dedup by content hash
        seen = set()
        unique_items: List[Dict[str, Any]] = []
        for item in pool_items:
            content = item.get("content", "") or ""
            h = fast_hash(content)
            if h in seen:
                continue
            seen.add(h)
            unique_items.append(item)

        if not unique_items:
            return []

        # 3) Rerank in batches (keep order stable)
        scores = self._rerank(query, unique_items, batch_size)

        # 4) Top-k selection without sorting everything
        # zip keeps alignment; nlargest picks the k best by score
        top_pairs = heapq.nlargest(k, zip(unique_items, scores), key=lambda x: x[1])
        return [item for item, _ in top_pairs]

    def _rerank(self, query: str, items: List[Dict[str, Any]], batch_size: int) -> List[float]:
        """
        Build (query, doc) pairs and get scores in batches.
        """
        pairs: List[Tuple[str, str]] = [(query, (it.get("content", "") or "")) for it in items]
        out: List[float] = []
        for batch in batch_iterable(pairs, batch_size):
            try:
                scores = self.rerank.predict(batch)
                # Ensures we extend with the same count; if the reranker misbehaves, pad zeros
                if not isinstance(scores, list) or len(scores) != len(batch):
                    raise ValueError("rerank.predict returned unexpected shape")
                out.extend(scores)
            except Exception as e:
                print(f"[RAG] Reranker batch error: {e}")
                out.extend([0.0] * len(batch))
        return out
