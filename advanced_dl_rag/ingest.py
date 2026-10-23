
import json
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

import pymupdf4llm
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT.parent

PDF_PATH = PROJECT_ROOT / "data" / "dl.pdf"
INDEX_DIR = BASE_DIR / "indexes"
CHROMA_DIR = INDEX_DIR / "chroma_cleaned"
PARENT_PATH = INDEX_DIR / "parents" / "parents_cleaned.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "advanced_dl_child_chunks_cleaned"

PARENT_CHUNK_SIZE = 1500
PARENT_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 80


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line).strip()


def extract_pages():
    print("Extracting PDF with PyMuPDF4LLM...")
    page_chunks = pymupdf4llm.to_markdown(
        str(PDF_PATH),
        page_chunks=True,
    )

    pages = []

    for index, page_chunk in enumerate(page_chunks):
        text = clean_text(page_chunk.get("text", ""))

        if not text:
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(PDF_PATH),
                    "page": index + 1,
                    "page_label": str(index + 1),
                },
            )
        )

    if not pages:
        raise ValueError("No readable text was extracted from the PDF.")

    print(f"Pages extracted: {len(pages):,}")
    print("\nExtraction preview:\n")
    print(pages[min(204, len(pages) - 1)].page_content[:2000])

    return pages


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if CHROMA_DIR.exists():
        raise FileExistsError(
            f"Clean index directory already exists: {CHROMA_DIR}\n"
            "Rename or back it up before running ingestion again."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PARENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pages = extract_pages()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_OVERLAP,
        add_start_index=True,
    )

    parent_docs = parent_splitter.split_documents(pages)

    parent_records = []
    children = []

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        add_start_index=True,
    )

    for parent in parent_docs:
        parent_id = str(uuid4())

        parent_metadata = dict(parent.metadata)
        parent_metadata["parent_id"] = parent_id

        parent_records.append({
            "parent_id": parent_id,
            "text": parent.page_content,
            "metadata": parent_metadata,
        })

        for child in child_splitter.split_documents([parent]):
            child_id = str(uuid4())

            child.metadata = {
                "child_id": child_id,
                "parent_id": parent_id,
                "source": str(PDF_PATH),
                "page": int(parent.metadata.get("page", 0)),
                "page_label": str(
                    parent.metadata.get(
                        "page_label",
                        parent.metadata.get("page", 0),
                    )
                ),
                "parent_start_index": int(
                    parent.metadata.get("start_index", 0)
                ),
                "child_start_index": int(
                    child.metadata.get("start_index", 0)
                ),
                "total_pages": len(pages),
            }

            children.append(child)

    print(f"\nParent passages: {len(parent_records):,}")
    print(f"Child chunks: {len(children):,}")

    print("Saving cleaned parent passages...")
    temporary_parent_path = PARENT_PATH.with_suffix(".tmp")

    with temporary_parent_path.open("w", encoding="utf-8") as file:
        for record in parent_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary_parent_path.replace(PARENT_PATH)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Creating separate cleaned child-chunk index...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    batch_size = 128

    for start in range(0, len(children), batch_size):
        batch = children[start:start + batch_size]

        vector_store.add_documents(
            documents=batch,
            ids=[doc.metadata["child_id"] for doc in batch],
        )

        print(
            f"Indexed {min(start + batch_size, len(children)):,}"
            f"/{len(children):,} child chunks"
        )

    print("\nClean indexing complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored child chunks: {vector_store._collection.count():,}")
    print(f"Parent file: {PARENT_PATH}")
    print(f"Vector index: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
