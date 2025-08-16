# 🤖 AdvRAG (Advanced Retrieval-Augmented Generation)

AdvRAG is a **modular RAG framework** built for **AI agent systems**.  
It gives you **total control** over ingestion, storage, retrieval, and reranking — so each agent can have its own knowledge base optimized for its task.  

---

## 🚀 Why AdvRAG?

Typical RAG pipelines are either **too rigid** (fixed configs, slow) or **too shallow** (no hybrid retrieval).  
AdvRAG fixes this by providing:

- **Multiple Config Presets** → `QualityConfig`, `MediumConfig`, `FastConfig`
- **Agent-oriented Knowledge Bases** → each agent gets its own persistent store
- **Hybrid Search** → dense embeddings + BM25 keyword search
- **Cross-Encoder Reranking** → ensures top answers are most relevant
- **File & Web Ingestion** → PDFs, CSV, Excel, Markdown, TXT, URLs
- **Caching** → faster repeated queries
- **Agentic-ready** → easily plug into **multi-agent frameworks**

---
## Use Case

Agent with its own knowledge base
```
from adv_rag import KnowledgeBase, AdvancedRAG, QualityConfig

# Create a knowledge base for Agent Alice
cfg = QualityConfig
kb = KnowledgeBase(
    files=["data/alice_notes.txt"],
    name="agent_alice",
    cfg=cfg
)

rag = AdvancedRAG(kb, cfg=cfg)

query = "What does Alice know about deployment?"
results = rag.retrieve(query, top_k=3)

print(rag.format(results))

```
Multi-agent knowledge bases

```
from adv_rag import KnowledgeBase, AdvancedRAG, MediumConfig, FastConfig

# Alice has high-quality knowledge storage
alice_kb = KnowledgeBase(dirs=["data/alice_docs"], name="alice", cfg=MediumConfig)
alice_rag = AdvancedRAG(alice_kb, cfg=MediumConfig)

# Bob needs fast lookups
bob_kb = KnowledgeBase(links=["https://example.com"], name="bob", cfg=FastConfig)
bob_rag = AdvancedRAG(bob_kb, cfg=FastConfig)

# Agents can query their own KBs
print("Alice:", alice_rag.format(alice_rag.retrieve("project roadmap")))
print("Bob:", bob_rag.format(bob_rag.retrieve("latest updates")))

```

Use inside an agentic framework
```
class MyAgent:
    def __init__(self, name, kb, rag):
        self.name = name
        self.kb = kb
        self.rag = rag

    def answer(self, query):
        docs = self.rag.retrieve(query)
        return f"{self.name} says:\n{self.rag.format(docs)}"


# Setup agents
alice = MyAgent("Alice", alice_kb, alice_rag)
bob = MyAgent("Bob", bob_kb, bob_rag)

# Ask them things
print(alice.answer("How do we deploy the system?"))
print(bob.answer("What’s trending in the docs?"))

```

## Covers

- File parsing (PDF, CSV, TXT, Excel)
- Web scraping
- Knowledge base ingestion
- RAG retrieval + reranking
