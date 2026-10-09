from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf():
    BASE_DIR = Path(__file__).resolve().parent.parent
    pdf_path = BASE_DIR / "data" / "Stock_Market_Performance_2024.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    return pages