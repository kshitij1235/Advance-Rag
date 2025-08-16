from adv_rag import KnowledgeBase, FastConfig
from pathlib import Path


def test_kb_ingestion(tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text("This is a test document about machine learning.")

    kb = KnowledgeBase(files=[str(txt)], name="testkb", cfg=FastConfig)
    assert kb.coll.count() > 0


def test_kb_hybrid_query(tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text("Artificial Intelligence and Neural Networks.")
    kb = KnowledgeBase(files=[str(txt)], name="testkb2", cfg=FastConfig)

    results = kb.hybrid("What about neural networks?", k=2, pool=3)
    assert len(results) > 0
