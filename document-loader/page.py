from langchain_community.document_loaders import WebBaseLoader
url = "https://www.geeksforgeeks.org/nlp/rag-architecture/"
data = WebBaseLoader(url)

docs=data.load()
print(docs[0].page_content[:1000])
print(docs[0].metadata)