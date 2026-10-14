from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
loader = PyPDFLoader("IJRTI2304061.PDF")
docs = loader.load()
spliter = TokenTextSplitter(
    chunk_size = 100,
    chunk_overlap=10
)
chunks = spliter.split_documents(docs)
print(chunks[0])