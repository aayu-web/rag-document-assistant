
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="deep_learning_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


CANDIDATE_K = 20
FINAL_K = 4

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

questions = [
    "When did deep learning begin?",
    "How are machine learning and deep learning related?",
    "What are the main waves in the history of deep learning?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    candidates = vector_store.similarity_search(
        question,
        k=CANDIDATE_K
    )

    pairs = [
        (question, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    # Rank candidates by cross-encoder score, highest first
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: float(item[0]),
        reverse=True
    )

    print("\n--- TOP RERANKED PASSAGES ---")

    for rank, (score, doc) in enumerate(
        ranked[:FINAL_K], start=1
    ):
        page = doc.metadata.get("page")
        page_label = doc.metadata.get("page_label")

        page_number = (
            page_label
            if page_label is not None
            else page + 1 if page is not None
            else "Unknown"
        )

        print(
            f"\n{rank}. Score: {float(score):.4f}"
            f" | PDF page: {page_number}"
        )
        print(doc.page_content[:500].strip())
