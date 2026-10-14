from langchain_community.document_loaders import PyPDFLoader

loader  = PyPDFLoader("document-loader/dl.pdf")
documents = loader.load()
print("Total pages",len(documents))
print("first page content")
print(documents[0].page_content[:1000])
print("\n first page meta data")
print(documents[0].metadata)