
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="deep_learning_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

questions = [
    "When did deep learning begin?",
    "How are machine learning and deep learning related?",
    "What are the main waves in the history of deep learning?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    for name, retriever in [
        ("SIMILARITY SEARCH", similarity_retriever),
        ("MMR SEARCH", mmr_retriever)
    ]:
        print(f"\n--- {name} ---")

        docs = retriever.invoke(question)

        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = doc.metadata.get("page_label")
            page_number = (
                page_label
                if page_label is not None
                else page + 1 if page is not None
                else "Unknown"
            )

            print(f"\n{i}. PDF page {page_number}")
            print(doc.page_content[:350].strip())
