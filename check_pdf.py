
from rag.loader import load_pdf

pages = load_pdf()

for page_number, page in enumerate(pages, start=1):
    text = page.page_content
    if "nvidia" in text.lower():
        print(f"\n--- PAGE {page_number} ---")

        # Print only the parts around Nvidia
        text_lower = text.lower()
        start = 0

        while True:
            index = text_lower.find("nvidia", start)
            if index == -1:
                break

            print(text[max(0, index - 300):index + 1200])
            print("-" * 50)
            start = index + len("nvidia")