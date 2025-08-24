from adv_rag import KnowledgeBase,AdvancedRAG,CompanyConfig

def run():
    kb = KnowledgeBase(links=["https://www.amazon.in/MSALA-DEVICE-Insert-Breathable-Sleeper/dp/B0CNM2XL4H"],cfg=CompanyConfig)
    rag = AdvancedRAG(kb)

    results = rag.retrieve("what is secrete code ", top_k=3)
    print(rag.format(results))


if __name__ == "__main__":
    run()
