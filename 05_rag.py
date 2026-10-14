import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma(
    collection_name="deep_learning_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0
)

def format_docs(docs):
    """format retrived chunks with their source"""
    formatted_docs=[]
    for doc in docs:
        source = doc.metadata.get("source","unkown source")
        page = doc.metadata.get("page")
        page_number = page +1 if page is not None else None
        citation = f"source: {source}"
        if page_number is not None:
            citation += f", PDF page {page_number}"

        formatted_docs.append(
            f"{doc.page_content}\n[{citation}]"
        )

    return "\n\n".join(formatted_docs)
prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant answering questions from documents.

Instructions:
1. Answer using only the supplied context.
2. Do not invent facts or sources.
3. If the context does not contain the answer, say:
   "I could not find the answer in the provided document."
4. Cite the source and PDF page number for factual claims.
5. Use the citation information supplied in the context.
6. Keep answers clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")

rag_chain = (
    {
        "context": retriver | format_docs,
        "question":RunnablePassthrough()


    }
    |prompt
    |llm
    |StrOutputParser()
)
def get_sources(question):
    """Retrieve and deduplicate source passages."""
    docs = retriver.invoke(question)

    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")

        
        page_label = doc.metadata.get("page_label")
        page = doc.metadata.get("page")

        if page_label:
            page_number = str(page_label)
        elif page is not None:
            page_number = str(page + 1)
        else:
            page_number = "Unknown"

        key = (source, page_number)

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "page": page_number,
            "excerpt": doc.page_content[:300].strip()
        })

    return sources

while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:
        answer = rag_chain.invoke(question)
        sources = get_sources(question)

        print("\nAnswer:\n", answer)
        print("\nSources:")

        if not sources:
            print("No source passages retrieved.")
        else:
            for i, source in enumerate(sources, start=1):
                print(f"\n[{i}] {source['source']}")

                if source["page"] is not None:
                    print(f"PDF page: {source['page']}")

                print("Excerpt:", source["excerpt"])

    except Exception as e:
        print(f"\nError: {e}")
