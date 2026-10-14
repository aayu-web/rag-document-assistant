
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load PDF
loader = PyPDFLoader("document-loader/dl.pdf")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)

chunks = splitter.split_documents(documents)
print("Total chunks:", len(chunks))

# Step 3: Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create persistent vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="deep_learning_docs",
    persist_directory="./chroma_db"
)

print("Stored vectors:", vector_store._collection.count())

# Step 5: Test semantic search
question = "What is deep learning?"

results = vector_store.similarity_search(
    question,
    k=3
)

print("\nRetrieved passages:")
for i, result in enumerate(results, start=1):
    print(f"\nPassage {i}:")
    print(result.page_content)
    print("Metadata:", result.metadata)
