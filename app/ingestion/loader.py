"""
loader.py

Responsible for:
1. Loading raw documents (PDF, for now) from disk.
2. Splitting them into overlapping text chunks suitable for embedding.

We keep this file dumb and single-purpose on purpose: it doesn't know about
embeddings, the database, or Gemini. That separation lets us test/debug
chunking quality completely independently of API calls or DB writes.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    """
    Loads a PDF and returns a list of LangChain Document objects,
    one per page, each with .page_content (text) and .metadata (source, page number).
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def chunk_documents(documents, chunk_size: int = 1200, chunk_overlap: int = 150):
    """
    Splits loaded documents into overlapping chunks.

    chunk_size=1200 characters is a practical balance for this app: fewer
    chunks means faster upload/indexing, while still keeping each passage
    focused enough for retrieval. We're using character count here (not
    token count) for simplicity; LangChain's splitter operates on characters
    by default unless you give it a token-counting function.

    chunk_overlap=150 means each chunk repeats the last 150 characters of the
    previous chunk. This prevents a sentence or idea from being cut cleanly
    in half at a chunk boundary, which would otherwise destroy context for
    whichever half gets retrieved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        # ^ tries to split on paragraph breaks first, then lines, then
        # sentences, then words, then characters as a last resort.
        # This ordering is *why* it's called "recursive" — it recursively
        # tries coarser-to-finer separators until chunks fit chunk_size.
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check chunking
    # on your test PDF before wiring up embeddings or the database.
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/test.pdf"

    print(f"Loading: {test_path}")
    pages = load_pdf(test_path)
    print(f"Loaded {len(pages)} page(s).")

    chunks = chunk_documents(pages)
    print(f"Split into {len(chunks)} chunk(s).\n")

    print("--- First chunk preview ---")
    print(chunks[0].page_content)
    print("\n--- Metadata ---")
    print(chunks[0].metadata)