import re

try:
    import tiktoken
except:
    tiktoken = None

try:
    import torch
except:
    torch = None


def _get_tokenizer():
    if tiktoken:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    class FallbackTokenizer:
        def encode(self, s: str):
            return s.split()

        def decode(self, t):
            return " ".join(t)

    return FallbackTokenizer()


def chunk_text(text: str, size: int, overlap: int):
    tok = _get_tokenizer()
    tokens = tok.encode(text)
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(tokens), step):
        window = tokens[i:i + size]
        if not window:
            break
        try:
            ch = tok.decode(window)
        except Exception:
            ch = " ".join(window)
        out.append(ch)
    return out or [text]


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9\-_\.]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "default"


def _device():
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"
