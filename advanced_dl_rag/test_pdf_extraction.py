from pathlib import Path
import pymupdf
import pymupdf4llm

pdf_path = Path(__file__).parent / "data" / "dl.pdf"

doc = pymupdf.open(pdf_path)

print("\n--- PyMuPDF extraction ---\n")
print(doc[204].get_text("text", sort=True)[:2500])

print("\n--- PyMuPDF4LLM extraction ---\n")
markdown = pymupdf4llm.to_markdown(
    str(pdf_path),
    pages=[204]
)
print(markdown[:2500])