# RefineRAG

RefineRAG is a small Retrieval-Augmented Generation demo built with a FastAPI backend, LangGraph agent flow, Chroma vector storage, and a Streamlit frontend.

It includes three agents:

- Retriever Agent: finds relevant document chunks for a query.
- Critique Agent: checks retrieval quality and answer grounding.
- Generator Agent: writes the final answer from approved context.

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

## Environment Variables

`.env.example` contains the supported settings:

- `GOOGLE_API_KEY`
- `GEMINI_MODEL`
- `EMBEDDING_MODEL`
- `VECTOR_STORE_PATH`
- `MAX_RETRIEVAL_RETRIES`
- `TOP_K`
- `BACKEND_URL`

## How It Works

1. Upload a document in Streamlit.
2. The frontend sends it to `POST /ingest`.
3. The backend loads, splits, embeds, and stores the chunks in Chroma.
4. A question goes to `POST /ask`.
5. LangGraph runs retrieval, critique, generation, and answer critique.
6. The frontend shows the answer, sources, and critique.

## API Endpoints

- `GET /health`: health check.
- `POST /ingest`: upload a PDF, TXT, or Markdown file.
- `POST /ask`: ask a question and receive answer, sources, and critique.
- `POST /reset-index`: clear the local vector store and uploaded raw files.

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

