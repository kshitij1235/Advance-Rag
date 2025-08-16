from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_model_name: str
    reranker_model_name: str
    chroma_path: str = "database"
    collection_name: str = "knowledge_base"
    bm25_enable: bool = True
    top_k_initial: int = 5
    top_k_rerank_pool: int = 10


# 🔹 High quality (slower, but best accuracy)
QualityConfig = RAGConfig(
    chunk_size_tokens=300,
    chunk_overlap_tokens=50,
    embedding_model_name="BAAI/bge-large-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-large",
    top_k_initial=8,
    top_k_rerank_pool=20,
)

# 🔹 Balanced (medium speed/accuracy)
MediumConfig = RAGConfig(
    chunk_size_tokens=400,
    chunk_overlap_tokens=80,
    embedding_model_name="BAAI/bge-base-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-base",
    top_k_initial=5,
    top_k_rerank_pool=10,
)

# 🔹 Fastest (for speed / prototyping)
FastConfig = RAGConfig(
    chunk_size_tokens=600,
    chunk_overlap_tokens=100,
    embedding_model_name="BAAI/bge-small-en-v1.5",
    reranker_model_name="BAAI/bge-reranker-base", 
    top_k_initial=3,
    top_k_rerank_pool=6,
)
