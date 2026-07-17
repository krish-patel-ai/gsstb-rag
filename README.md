cd ~/Desktop/gsstb-rag

cat > README.md << 'EOF'
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
- LangGraph pipeline: query rewriting (for follow-ups) -> routing -> retrieval -> generation
- Grounded refusal: a consistent, exact refusal string used both when
  retrieval finds nothing relevant and when the LLM itself can't answer
  from what it was given -- never silently hallucinates
- Multi-turn conversational memory, including standalone-question rewriting
  for ambiguous follow-ups (e.g. "explain it in more detail")
- Streaming responses in the Streamlit chat UI
- Dockerized deployment for easy setup and portability

## Setup

### Option A: Local (venv)

**1. Install dependencies**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Set your Groq API key**

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here

**3. Add textbook PDFs**

Download PDFs from the [GSSTB site](https://gsstb.gujarat.gov.in/Home/gsstb/Std9to12)
and place them following this folder convention:
data/pdfs/
<standard>/
<subject>/
yourfile.pdf

e.g. `data/pdfs/std10/computer/book.pdf`. Folder names become metadata used
by the router -- keep them lowercase and consistent.

**4. Build the index**
```bash
python3 ingestion/ingestion.py
```

Extracts text page-by-page from every PDF (falling back to OCR on pages
with no extractable text layer), chunks it at sentence/paragraph
boundaries, and writes both a ChromaDB vector index and a BM25 keyword
index to `data/chroma_db/`.

**5. Run the app**
```bash
streamlit run app/Streamlit_app.py
```

### Option B: Docker

```bash
docker build -t gsstb-rag .
docker run -p 8501:8501 --env-file .env gsstb-rag
```

Open `http://localhost:8501` in your browser. The Dockerfile installs a
CPU-only build of PyTorch (via PyTorch's dedicated CPU package index) to
keep the image size reasonable, since GPU acceleration isn't available in
a standard Docker Desktop environment on macOS.

## Project structure

```
gsstb-rag/
├── app/
│   └── Streamlit_app.py       # Chat UI with streaming + citations
├── data/
│   ├── chroma_db/               # generated ChromaDB + BM25 index
│   └── pdfs/                    # source textbook PDFs (not committed)
│       ├── std10/
│       └── std12/
├── ingestion/
│   ├── ingestion.py            # PDF -> page-tagged chunks -> ChromaDB + BM25
│   └── ocr.py                   # OCR fallback for scanned/image PDF pages
├── retrieval/
│   ├── graph.py                 # LangGraph: rewrite -> route -> retrieve -> generate
│   └── hybrid_retriever.py     # dense + sparse search, RRF fusion, reranking
├── screenshots/                 # example interactions (see below)
├── .dockerignore
├── .env                          # not committed (GROQ_API_KEY)
├── .gitignore
├── ARCHITECTURE.md               # design decisions and tradeoffs
├── Dockerfile
├── README.md
└── requirements.txt
```

## Example interactions to demo

- A question clearly answerable from one textbook (tests routing + citations)
- A question outside the indexed corpus (tests the refusal path)
- A follow-up question relying on earlier conversation context, including
  non-question follow-ups like "explain that in more detail" (tests query
  rewriting + multi-turn memory)

## Screenshots

See the `screenshots/` folder for example interactions covering:
- Clearly answerable questions with accurate citations (textbook name +
  page number + snippet)
- Conversational follow-ups (e.g. "explain it in more detail")
- The refusal path when a topic isn't covered by the indexed textbooks
- Answers spanning multiple textbooks and standards

## Known limitations

- Retrieval ranking can occasionally favor a passage that references a
  topic in passing over one that explicitly defines it, leading to a
  conservative refusal even when related content exists elsewhere. See
  `ARCHITECTURE.md` for a concrete, tested example.
- No authentication, session persistence, or systematic performance
  optimization-scoped out given time constraints in favor of
  retrieval-quality features.