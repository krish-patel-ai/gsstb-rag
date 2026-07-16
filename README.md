# GSSTB Conversational RAG System

A conversational AI assistant that answers questions strictly from Gujarat State
Board (GSSTB) textbooks, with page-accurate citations, hybrid retrieval, query
routing, and OCR support for scanned pages.

Built for the Kwiqwork AI/ML Internship practical assignment. See
`ARCHITECTURE.md` for design decisions and tradeoffs.

## Features

- Multi-textbook ingestion (any GSSTB standard/subject) with page-level citation tracking
- OCR fallback (RapidOCR) for pages with no extractable text layer
- Sentence-aware chunking that respects paragraph and sentence boundaries
  rather than blindly slicing by character count
- Hybrid search: dense embeddings (BAAI/bge-small-en-v1.5 via ChromaDB) +
  BM25 keyword search, fused via Reciprocal Rank Fusion
- Cross-encoder reranking for final retrieval precision
- LangGraph pipeline: query rewriting (for follow-ups) → routing → retrieval → generation
- Grounded refusal: a consistent, exact refusal string used both when
  retrieval finds nothing relevant and when the LLM itself can't answer
  from what it was given — never silently hallucinates
- Multi-turn conversational memory, including standalone-question rewriting
  for ambiguous follow-ups (e.g. "explain it in more detail")
- Streaming responses in the Streamlit chat UI

## Setup

### 1. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your Groq API key

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here

### 3. Add textbook PDFs

Download PDFs from the [GSSTB site](https://gsstb.gujarat.gov.in/Home/gsstb/Std9to12)
and place them following this folder convention:

data/pdfs/
<standard>/
<subject>/
yourfile.pdf

e.g. `data/pdfs/std10/science/book.pdf`. Folder names become metadata used
by the router — keep them lowercase and consistent.

### 4. Build the index

```bash
python3 ingestion/ingestion.py
```

Extracts text page-by-page from every PDF (falling back to OCR on pages
with no extractable text layer), chunks it at sentence/paragraph
boundaries, and writes both a ChromaDB vector index and a BM25 keyword
index to `data/chroma_db/`.

### 5. Run the app

```bash
streamlit run app/Streamlit_app.py
```

## Project structure
gsstb-rag/
├── ingestion/
│   ├── ingestion.py         # PDF -> page-tagged chunks -> ChromaDB + BM25
│   └── ocr.py                # OCR fallback for scanned/image PDF pages
├── retrieval/
│   ├── hybrid_retriever.py  # dense + sparse search, RRF fusion, reranking
│   └── graph.py               # LangGraph: rewrite -> route -> retrieve -> generate
├── app/
│   └── Streamlit_app.py     # Chat UI with streaming + citations
├── data/
│   ├── pdfs/                  # source textbook PDFs (not committed)
│   └── chroma_db/             # generated index (not committed)
├── ARCHITECTURE.md            # design decisions and tradeoffs
└── requirements.txt

## Example interactions to demo

- A question clearly answerable from one textbook (tests routing + citations)
- A question outside the indexed corpus (tests the refusal path)
- A follow-up question relying on earlier conversation context, including
  non-question follow-ups like "explain that in more detail" (tests query
  rewriting + multi-turn memory)

## Known limitations

- Chunking respects sentence/paragraph boundaries, but ranking quality can
  still vary depending on whether a topic is explicitly defined in the
  indexed pages versus only referenced in passing.
- No Dockerized deployment, authentication, or session persistence —
  scoped out given time constraints in favor of retrieval-quality features.
