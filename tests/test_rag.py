from adv_rag import KnowledgeBase, AdvancedRAG, FastConfig
from pathlib import Path


def test_rag_retrieve(tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text("Python is a programming language.")
    
    kb = KnowledgeBase(files=[str(txt)], name="testrag", cfg=FastConfig)
    rag = AdvancedRAG(kb, cfg=FastConfig)

    results = rag.retrieve("What is Python?", top_k=1)
    assert len(results) > 0
    assert "Python" in results[0]["content"]
