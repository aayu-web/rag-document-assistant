
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="deep_learning_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

print("Stored chunks:", vector_store._collection.count())



reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

CANDIDATE_K = 20
FINAL_K = 4



prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant answering questions
from the provided textbook passages.

Rules:
- Answer using the provided context only.
- Do not invent facts that are not supported by the context.
- If the context does not contain the answer, say so clearly.
- Explain the answer in clear, simple language.
- Refer to the source page numbers when useful.

Context:
{context}"""
    ),
    ("human", "{question}"),
])



def get_page_number(doc):
    page_label = doc.metadata.get("page_label")

    if page_label is not None and str(page_label).strip():
        return str(page_label)

    page = doc.metadata.get("page")

    if page is not None:
        return str(int(page) + 1)

    return "Unknown"


def retrieve_and_rerank(question):
    
    candidates = vector_store.similarity_search(
        question,
        k=CANDIDATE_K,
    )

    if not candidates:
        return []

    
    pairs = [
        (question, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    return ranked[:FINAL_K]




def ask_question(question):
    ranked_docs = retrieve_and_rerank(question)

    if not ranked_docs:
        print("No relevant passages were found.")
        return

    context_parts = []

    for rank, (score, doc) in enumerate(ranked_docs, start=1):
        page = get_page_number(doc)

        context_parts.append(
            f"[Source {rank} | PDF page {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = prompt.invoke({
        "question": question,
        "context": context,
    })

    response = llm.invoke(messages)

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    print("\nANSWER:\n")
    print(response.content)

    print("\n" + "-" * 70)
    print("SOURCES USED")
    print("-" * 70)

    seen = set()

    for rank, (score, doc) in enumerate(ranked_docs, start=1):
        page = get_page_number(doc)
        source = doc.metadata.get("source", "Unknown source")
        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        print(
            f"\nSource {rank} | PDF page {page}"
            f" | Reranker score: {float(score):.4f}"
        )
        print("File:", source)
        print("Excerpt:", doc.page_content[:350].strip())




if __name__ == "__main__":
    print("\nDeep Learning RAG Assistant")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            ask_question(question)
        except Exception as error:
            print(f"\nAn error occurred: {error}")
