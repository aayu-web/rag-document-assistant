
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("document-loader/dl.pdf")
documents = loader.load()

print("Original page documents:", len(documents))

# Step 2: Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)

# Step 3: Split documents into chunks
chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Step 4: Inspect the first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content)

    # Step 5: Inspect metadata
    print("\nFirst chunk metadata:")
    print(chunks[0].metadata)

    # Step 6: Inspect chunk lengths
    print("\nFirst 5 chunk lengths:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i + 1}: {len(chunk.page_content)} characters")
else:
    print("No text chunks were created.")
