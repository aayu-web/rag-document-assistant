from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("document-loader/dl.pdf")
document = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=100,
    add_start_index=True
)

chunks = splitter.split_documents(document)

print("Total chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)

print("total embeddings: ",len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("First vector's first 10 values:", vectors[0][:10])

assert len(vectors) == len(chunks)

print("Every chunk has an embedding!")