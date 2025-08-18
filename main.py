from adv_rag import KnowledgeBase,AdvancedRAG,CompanyConfig

def run():
    kb = KnowledgeBase(dirs=["knowledge/docs"], name="chem",cfg=CompanyConfig)
    rag = AdvancedRAG(kb)

    results = rag.retrieve("what is secrete code ", top_k=3)
    print(rag.format(results))


if __name__ == "__main__":
    run()
