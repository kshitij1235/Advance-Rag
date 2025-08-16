from adv_rag import KnowledgeBase,AdvancedRAG,MediumConfig

def run():
    kb = KnowledgeBase(dirs=["knowledge/docs"], name="alice",cfg=MediumConfig)
    rag = AdvancedRAG(kb)

    results = rag.retrieve("what does alice know about deployment?", top_k=3)
    print(rag.format(results))


if __name__ == "__main__":
    run()
