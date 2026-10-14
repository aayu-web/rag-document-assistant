#load the data 
#split the data into small chunks 
#create the embeddings 
#store into chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("document-loader/dl.pdf")
docs = loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)


chunks = spliter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
print("Documents in database:", vectorstore._collection.count())



