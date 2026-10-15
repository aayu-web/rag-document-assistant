
import os
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder




load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "deep_learning_docs"

CANDIDATE_K = 20
FINAL_K = 4




print("\n" + "=" * 70)
print("INITIALIZING RAG CHATBOT")
print("=" * 70)

startup_start = time.perf_counter()

print("\n[1/3] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("[2/3] Connecting to Chroma vector store...")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

print(f"Stored chunks: {vector_store._collection.count():,}")




print("[3/3] Loading CrossEncoder reranker...")

reranker = CrossEncoder(RERANKER_MODEL)

startup_time = time.perf_counter() - startup_start

print(f"\nStartup time: {startup_time:.2f} seconds")
print("Chatbot is ready.")




api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Add GROQ_API_KEY=your_key to your .env file."
    )

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    api_key=api_key,
)




PROMPT = ChatPromptTemplate.from_template("""
You are a careful and helpful document-based question-answering assistant.

Answer the user's question using only the supplied context.

Instructions:
- Give a clear and accurate answer.
- Use simple language.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say so clearly.
- Cite supporting information using the provided source references.
- Preserve relevant technical details when necessary.
- Do not claim that a source says something it does not say.

Context:
{context}

Question:
{question}

Answer:
""")




def get_page_number(doc):
    """Return a human-readable PDF page number when available."""

    page_label = doc.metadata.get("page_label")

    if page_label:
        return str(page_label)

    page = doc.metadata.get("page")

    if page is not None:
        return str(int(page) + 1)

    return "Unknown"




def retrieve_and_rerank(question):
    """
    Retrieve candidate chunks from Chroma and rerank them
    using the CrossEncoder.

    Returns:
        final_docs: Ranked list of (document, score) pairs.
        timings: Retrieval and reranking durations.
    """

    timings = {
        "retrieval": 0.0,
        "reranking": 0.0,
    }



    retrieval_start = time.perf_counter()

    candidates = vector_store.similarity_search(
        question,
        k=CANDIDATE_K,
    )

    timings["retrieval"] = (
        time.perf_counter() - retrieval_start
    )

    if not candidates:
        return [], timings

 

    reranking_start = time.perf_counter()

    pairs = [
        (question, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    final_docs = ranked_docs[:FINAL_K]

    timings["reranking"] = (
        time.perf_counter() - reranking_start
    )

    return final_docs, timings




def ask_question(question):
    """Run the complete RAG pipeline and display timings."""

    total_start = time.perf_counter()



    ranked_docs, timings = retrieve_and_rerank(question)

    if not ranked_docs:
        total_time = time.perf_counter() - total_start

        print("\nNo relevant document passages were found.")
        print(f"Total response time: {total_time:.3f} seconds")

        return


    context_parts = []

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for rank, (doc, score) in enumerate(ranked_docs, start=1):
        page = get_page_number(doc)
        source = os.path.basename(
            doc.metadata.get("source", "Document")
        )

        context_parts.append(
            f"[Source {rank} | File: {source} | PDF page: {page}]\n"
            f"{doc.page_content}"
        )

        print(
            f"\nSource {rank}: {source}"
            f"\nPDF page: {page}"
            f"\nReranker score: {float(score):.4f}"
        )

        excerpt = doc.page_content[:300].strip()
        print(f"Excerpt: {excerpt}...")

    context = "\n\n".join(context_parts)



    print("\nGenerating answer...")

    generation_start = time.perf_counter()

    response = llm.invoke(
        PROMPT.format_messages(
            question=question,
            context=context,
        )
    )

    timings["generation"] = (
        time.perf_counter() - generation_start
    )



    total_time = time.perf_counter() - total_start



    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(response.content)



    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)

    print(f"Retrieval time:       {timings['retrieval']:.3f} seconds")
    print(f"Reranking time:       {timings['reranking']:.3f} seconds")
    print(f"LLM generation time:  {timings['generation']:.3f} seconds")

    print("-" * 70)

    print(f"Total response time:  {total_time:.3f} seconds")

    print("=" * 70)




def main():
    print("\n" + "=" * 70)
    print("DEEP LEARNING DOCUMENT RAG CHATBOT")
    print("=" * 70)

    print("Ask questions about the indexed document.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nAsk a question: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("\nExiting chatbot. Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            ask_question(question)

        except KeyboardInterrupt:
            print("\n\nChatbot interrupted. Goodbye!")
            break

        except Exception as exc:
            print(f"\nAn error occurred: {exc}")





if __name__ == "__main__":
    main()
