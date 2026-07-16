import re
from pathlib import Path
from dataclasses import dataclass,asdict
import chromadb
import pickle
import fitz
from ocr import extract_page_with_ocr
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

PDF_ROOT = Path(__file__).parent.parent / "data" / "pdfs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
BM25_INDEX_PATH = CHROMA_DIR / "bm25_index.pkl"

COLLECTION_NAME = "gsstb_textbooks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE_CHARS=900
CHUNK_OVERLAP=150

@dataclass
class chunk:
    text:str
    textbook_name:str
    standard:str
    subject:str
    page_num:int
    chunk_id:str

def clean_text(text): # cleans the texts
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_page_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) < size:
            current += ("\n\n" if current else "") + paragraph
        else:
            if current:
                chunks.append(current)

            if len(paragraph) > size:
                chunks.extend(_split_long_text(paragraph, size, overlap))
                current = ""
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """Splits oversized text at sentence boundaries where possible,
    falling back to character slicing only if a single sentence
    itself exceeds the chunk size.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            if len(sentence) > size:
                start = 0
                while start < len(sentence):
                    end = start + size
                    chunks.append(sentence[start:end])
                    start += size - overlap
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks

def extract_chunks_from_pdf(pdf_path:Path,standard:str,subject:str,textbook_name:str)->list[chunk]: # extracts text from a PDF and splits it into chunks
    chunks = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = clean_text(page.get_text())

            # OCR fallback for PDFs with no extractable text
            if not text:
                print(f"  OCR -> Page {page_num + 1}")
                text = clean_text(
                    extract_page_with_ocr(pdf_path, page_num)
                )

            if not text:
                continue
            page_chunks = chunk_page_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP)
            for i, page_chunk in enumerate(page_chunks):
                chunk_id = f"{pdf_path.stem}_page{page_num+1}_chunk{i+1}"
                chunks.append(chunk(text=page_chunk, textbook_name=textbook_name, standard=standard, subject=subject, page_num=page_num+1, chunk_id=chunk_id))
    return chunks

def discover_pdfs()-> list[tuple[Path, str, str]]:  # discovers PDFs in the data/pdfs directory and returns a list of tuples containing the PDF path, standard, and subject
    found=[]
    if not PDF_ROOT.exists():
        raise FileNotFoundError(
            f"{PDF_ROOT} does not exist. Create it and drop PDFs in "
            f"data/pdfs/<standard>/<subject>/*.pdf, e.g. data/pdfs/std10/science/book.pdf"
        )
    for standard_dir in sorted(PDF_ROOT.iterdir()):
        if not standard_dir.is_dir():
            continue
        for subject_dir in sorted(standard_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for pdf_file in sorted(subject_dir.glob("*.pdf")):
                found.append((pdf_file, standard_dir.name, subject_dir.name, pdf_file.stem))
    return found

def build_index(): # builds the index by extracting chunks from PDFs and storing them in ChromaDB and BM25 index
    pdf_files=discover_pdfs()
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_ROOT}. Drop PDFs in "
            f"data/pdfs/<standard>/<subject>/*.pdf, e.g. data/pdfs/std10/science/book.pdf"
        )
    all_chunks: list[chunk] = []
    for pdf_path, standard, subject, _ in pdf_files:
        textbook_name = f"{standard.upper()} {subject.title()}"
        print(f"Processing {pdf_path} for {textbook_name}")
        chunks = extract_chunks_from_pdf(pdf_path, standard, subject, textbook_name)
        print(f"  extracted {len(chunks)} chunks")
        all_chunks.extend(chunks)
        if not all_chunks:
            print("No text extracted from any PDF. Aborting.")
            return

    print(f"\nTotal chunks across all textbooks: {len(all_chunks)}")

# write the chunks to a ChromaDB collection
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)   
    client=chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        pass
    collection=client.create_collection(name=COLLECTION_NAME,embedding_function=embedding_function)
    BATCH_SIZE = 200
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i+BATCH_SIZE]
        ids = [c.chunk_id for c in batch_chunks]
        documents = [c.text for c in batch_chunks]
        metadatas = [{"textbook_name": c.textbook_name, "standard": c.standard, "subject": c.subject, "page_num": c.page_num} for c in batch_chunks]
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"Indexed {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks...")

    print(f"Wrote {len(all_chunks)} chunks to ChromaDB collection '{COLLECTION_NAME}'")

    tokenized_corpus = [
        re.findall(r"\b\w+\b", c.text.lower())
        for c in all_chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in all_chunks],
            "chunk_meta": {c.chunk_id: asdict(c) for c in all_chunks},
        }, f)
    print(f"Wrote BM25 index to {BM25_INDEX_PATH}")

    print("\nIngestion complete.")

if __name__ == "__main__":
    build_index()
