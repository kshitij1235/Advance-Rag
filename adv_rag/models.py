from sentence_transformers import SentenceTransformer, CrossEncoder
from .utils import _device


def get_embedder(model_name: str):
    return SentenceTransformer(model_name, device=_device())


def get_reranker(model_name: str):
    return CrossEncoder(model_name, device=_device())
