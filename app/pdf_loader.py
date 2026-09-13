from pypdf import PdfReader
from app.preprocessing import clean_text

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Returns a list like:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        pages.append({
            "page": page_number,
            "text": text
        })
    return pages