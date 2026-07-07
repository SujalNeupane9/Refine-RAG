# RefineRAG

RefineRAG is a small Retrieval-Augmented Generation demo built with a FastAPI backend, a LangGraph workflow, Chroma vector storage, and a Streamlit frontend.

The system is designed to:

- ingest local documents
- split them into chunks
- store the chunks in a vector database
- retrieve relevant context for a user question
- generate an answer from that context
- show sources and a lightweight critique in the UI

It currently uses three main agents:

- Retriever Agent: finds relevant document chunks for a query.
- Critique Agent: checks retrieval quality and answer grounding.
- Generator Agent: writes the final answer from retrieved context.

## Project Layout

```text
RefineRAG/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── graph/
│   │   ├── rag/
│   │   └── services/
│   ├── data/
│   └── tests/
├── frontend/
├── docker-compose.yml
├── .env.example
├── README.md
└── change_log.md
```

## Requirements

- Python 3.11+
- Docker and Docker Compose for containerized runs
- A Gemini API key for the main LLM workflow

If `GOOGLE_API_KEY` is missing, the app can still fall back to local hash embeddings for retrieval, but the generator and critique steps expect the Gemini API key to be present.

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install backend and frontend dependencies.

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

3. Create your local environment file.

```bash
cp .env.example .env
```

4. Edit `.env` and set `GOOGLE_API_KEY`.

The app reads `.env` for local runs, and Docker Compose also uses it when available.

## Environment Variables

`.env.example` contains the supported settings:

- `GOOGLE_API_KEY`: required for Gemini generation and critique
- `GEMINI_MODEL`: model used for generation and critique
- `EMBEDDING_MODEL`: model used for Gemini embeddings when available
- `VECTOR_STORE_PATH`: location of the local Chroma store
- `MAX_RETRIEVAL_RETRIES`: number of retrieval retries before the graph proceeds
- `TOP_K`: number of chunks retrieved per query
- `BACKEND_URL`: backend URL used by the frontend

Default values are already wired into the code, so only `GOOGLE_API_KEY` is strictly required for the full Gemini-backed workflow.

## Run Locally

Start the backend from the `backend/` directory:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, start the frontend from the project root:

```bash
cd ..
streamlit run frontend/app.py
```

Open:

- Backend health: `http://localhost:8000/health`
- Frontend: `http://localhost:8501`

Local runtime notes:

- The backend stores uploaded files under `backend/data/raw/`.
- The vector store defaults to `backend/data/vector_store/`.
- If Gemini embeddings cannot be created, the backend falls back to local hash embeddings.

## Run With Docker

Build and start both services:

```bash
docker compose up --build
```

Stop the stack:

```bash
docker compose down
```

Docker exposes:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`
- Docker Compose reads `.env` if it exists.
- The backend container also gets sensible defaults for most settings.
- The frontend defaults to `http://backend:8000` when no `BACKEND_URL` is set.

## How It Works

1. Upload a document in Streamlit.
2. The frontend sends it to `POST /ingest`.
3. The backend loads, splits, embeds, and stores the chunks in Chroma.
4. A question goes to `POST /ask`.
5. LangGraph runs retrieval, retrieval critique, generation, and answer critique.
6. The frontend shows the answer, sources, and critique.

### Workflow Details

- Retrieval starts by searching the vector store for the most relevant chunks.
- The critique step can ask for a refined query when retrieval quality looks weak.
- The generator only answers from retrieved context and should not invent facts.
- If the answer is weak, the workflow allows one regeneration pass.

The current critique step is intentionally lightweight and mostly used to support retry decisions. It is not a full evaluation framework.

## API Endpoints

- `GET /health`: health check.
- `POST /ingest`: upload a PDF, TXT, or Markdown file.
- `POST /ask`: ask a question and receive answer, sources, and critique.
- `POST /reset-index`: clear the local vector store and uploaded raw files.

### `POST /ingest`

Uploads a document and writes it to the backend raw file directory before chunking and indexing it.

Supported file types:

- PDF
- TXT
- Markdown

### `POST /ask`

Accepts JSON in the form:

```json
{"question":"What is RefineRAG?"}
```

Returns:

- `answer`
- `sources`
- `critique`

### `POST /reset-index`

Clears the local Chroma store and removes uploaded raw files from the backend data directory.

Example ingest request:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@backend/data/raw/sample_refinerag.md;type=text/markdown"
```

Example ask request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What vector store does RefineRAG use for local storage?"}'
```

## Development Notes

- The backend uses `python-dotenv` to load environment variables from `.env`.
- The frontend also calls `load_dotenv()` so local runs can share the same configuration.
- The code has fallbacks for some failure modes, but the Gemini key is still needed for the full experience.
- The `/reset-index` route is mainly a development convenience.