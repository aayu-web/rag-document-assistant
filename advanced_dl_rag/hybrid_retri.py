
import json
import os
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from src.prompts import QUESTION_REWRITE_PROMPT, ANSWER_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parent
RAG_ROOT = PROJECT_ROOT.parent

load_dotenv(RAG_ROOT / ".env")

CHROMA_DIR = RAG_ROOT / "indexes" / "chroma_cleaned"
PARENT_PATH = (
    RAG_ROOT / "indexes" / "parents" / "parents_cleaned.jsonl"
)

COLLECTION_NAME = "advanced_dl_child_chunks_cleaned"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

DENSE_K = 20
BM25_K = 20
FUSED_CHILD_K = 30
PARENT_RERANK_K = 20
FINAL_K = 5


def load_parents():
    parents = {}

    with PARENT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            parents[record["parent_id"]] = record

    print(f"Loaded parent passages: {len(parents):,}")
    return parents


def load_retrieval_components():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    stored = vector_store._collection.get(
        include=["documents", "metadatas"]
    )

    child_docs = []

    for text, metadata in zip(
        stored["documents"], stored["metadatas"]
    ):
        child_docs.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )

    if not child_docs:
        raise ValueError(
            "No child chunks found. Check your Chroma index."
        )

    print(f"Loaded child chunks: {len(child_docs):,}")

    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = BM25_K

    print("Loading cross-encoder reranker...")

    reranker = CrossEncoder(RERANKER_MODEL)

    return vector_store, bm25_retriever, reranker


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    documents = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            child_id = doc.metadata.get("child_id")

            if not child_id:
                continue

            scores[child_id] += 1.0 / (k + rank)
            documents[child_id] = doc

    fused = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents[child_id], score)
        for child_id, score in fused
    ]


def retrieve_candidate_parents(
    query,
    vector_store,
    bm25_retriever,
    parents,
):
    dense_results = vector_store.similarity_search(
        query,
        k=DENSE_K,
    )

    bm25_results = bm25_retriever.invoke(query)

    print(
        f"Dense results: {len(dense_results)} | "
        f"BM25 results: {len(bm25_results)}"
    )

    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    fused_results = fused_results[:FUSED_CHILD_K]

    candidate_parents = {}
    best_child_scores = {}

    for child_doc, rrf_score in fused_results:
        parent_id = child_doc.metadata.get("parent_id")

        if not parent_id or parent_id not in parents:
            continue

        if (
            parent_id not in best_child_scores
            or rrf_score > best_child_scores[parent_id]
        ):
            best_child_scores[parent_id] = rrf_score
            candidate_parents[parent_id] = parents[parent_id]

    candidates = list(candidate_parents.values())

    candidates.sort(
        key=lambda parent: best_child_scores[parent["parent_id"]],
        reverse=True,
    )

    candidates = candidates[:PARENT_RERANK_K]

    print(f"Unique candidate parents: {len(candidates)}")

    return candidates


def rerank_parents(query, candidate_parents, reranker):
    if not candidate_parents:
        return []

    pairs = [
        (query, parent["text"][:2500])
        for parent in candidate_parents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for parent, score in zip(candidate_parents, scores):
        result = dict(parent)
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:FINAL_K]


def format_context(results):
    context_parts = []

    for index, parent in enumerate(results, start=1):
        metadata = parent.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        source = Path(str(source).replace("\\", "/")).name

        page = metadata.get(
            "page_label",
            metadata.get("page", "Unknown"),
        )

        context_parts.append(
            f"[Passage {index} | Source: {source} | Page: {page}]\n"
            f"{parent['text']}"
        )

    return "\n\n".join(context_parts)


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            f"GROQ_API_KEY not found. Check your .env file at {RAG_ROOT}"
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma index not found: {CHROMA_DIR}"
        )

    if not PARENT_PATH.exists():
        raise FileNotFoundError(
            f"Parent file not found: {PARENT_PATH}"
        )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
    )

    rewrite_chain = QUESTION_REWRITE_PROMPT | llm
    answer_chain = ANSWER_PROMPT | llm

    parents = load_parents()

    vector_store, bm25_retriever, reranker = (
        load_retrieval_components()
    )

    chat_history = []

    print("\nPrompt-engineered hybrid RAG is ready.")
    print("Enter 'clear' to reset conversation history.")
    print("Enter 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if question.lower() == "clear":
            chat_history.clear()
            print("Conversation history cleared.")
            continue

        if not question:
            continue

        history_text = "\n".join(
            f"User: {item['question']}\nAssistant: {item['answer']}"
            for item in chat_history[-5:]
        )

        try:
            rewritten_response = rewrite_chain.invoke({
                "chat_history": history_text or "No previous conversation.",
                "question": question,
            })

            rewritten_query = rewritten_response.content.strip()

            if not rewritten_query:
                rewritten_query = question

            print(f"\nRewritten query: {rewritten_query}")

            candidates = retrieve_candidate_parents(
                rewritten_query,
                vector_store,
                bm25_retriever,
                parents,
            )

            results = rerank_parents(
                rewritten_query,
                candidates,
                reranker,
            )

            if not results:
                print("\nNo matching passages were found.")
                continue

            context = format_context(results)

            answer_response = answer_chain.invoke({
                "context": context,
                "question": question,
            })

            answer = answer_response.content

            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(answer)

            print("\n" + "=" * 70)
            print("RETRIEVED SOURCES")
            print("=" * 70)

            for rank, parent in enumerate(results, start=1):
                metadata = parent.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                source = Path(str(source).replace("\\", "/")).name
                page = metadata.get(
                    "page_label",
                    metadata.get("page", "Unknown"),
                )

                print(
                    f"{rank}. {source} | Page: {page} | "
                    f"Reranker score: {parent['reranker_score']:.4f}"
                )

            chat_history.append({
                "question": question,
                "answer": answer,
            })

        except Exception as error:
            print(f"\nError processing question: {error}")


if __name__ == "__main__":
    main()
