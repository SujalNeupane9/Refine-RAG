import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import AskRequest, AskResponse, IngestResponse, ResetIndexResponse, Source
from app.services.rag_service import ask_question, ingest_document, reset_ingestion_state


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RefineRAG API")
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@app.get("/health")
def health_check():
    logger.info("Health check requested.")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    logger.info("Ask endpoint requested.")
    result = ask_question(payload.question)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**source) for source in result["sources"]],
        critique=result.get("critique"),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex}_{Path(file.filename).name}"
    destination = RAW_DATA_DIR / safe_name
    logger.info("Ingest endpoint received file: original=%s stored_as=%s", file.filename, safe_name)

    try:
        contents = await file.read()
        destination.write_bytes(contents)
        chunks_created = ingest_document(destination)
    except ValueError as exc:
        logger.warning("Ingest failed for file=%s error=%s", safe_name, exc)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingest failure for file=%s", safe_name)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc

    return IngestResponse(
        message="Document ingested successfully",
        chunks_created=chunks_created,
    )


@app.post("/reset-index", response_model=ResetIndexResponse)
def reset_index_endpoint():
    # Development-only helper for clearing the local Chroma store and uploaded raw files.
    logger.info("Reset index endpoint requested.")
    reset_ingestion_state(RAW_DATA_DIR)
    return ResetIndexResponse(message="Index reset successfully")
