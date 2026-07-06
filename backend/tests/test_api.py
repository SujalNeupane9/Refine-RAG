import sys
import asyncio
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app import main  # noqa: E402


def test_health_endpoint():
    assert main.health_check() == {"status": "ok"}


def test_ask_endpoint_returns_typed_response(monkeypatch):
    monkeypatch.setattr(
        main,
        "ask_question",
        lambda question: {
            "answer": f"Answer for {question}",
            "sources": [{"file": "sample.md", "page": 1, "chunk_id": "0"}],
            "critique": "short critique",
        },
    )

    result = main.ask_endpoint(main.AskRequest(question="What is RefineRAG?"))

    assert result.model_dump() == {
        "answer": "Answer for What is RefineRAG?",
        "sources": [{"file": "sample.md", "page": 1, "chunk_id": "0"}],
        "critique": "short critique",
    }


def test_ingest_endpoint_saves_file_and_returns_chunk_count(monkeypatch):
    captured = {}

    def fake_ingest_document(path):
        captured["path"] = path
        return 3

    monkeypatch.setattr(main, "ingest_document", fake_ingest_document)

    class FakeUpload:
        filename = "demo.md"

        async def read(self):
            return b"# Demo\nHello world"

    result = asyncio.run(main.ingest_endpoint(FakeUpload()))

    assert result.model_dump() == {
        "message": "Document ingested successfully",
        "chunks_created": 3,
    }
    assert captured["path"].name.endswith("_demo.md")
    assert captured["path"].exists()
    captured["path"].unlink(missing_ok=True)


def test_reset_index_endpoint_clears_raw_directory(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / ".gitkeep").write_text("")
    (raw_dir / "temp.txt").write_text("temp")

    captured = {}

    def fake_reset_ingestion_state(path):
        captured["path"] = Path(path)
        for child in Path(path).iterdir():
            if child.name != ".gitkeep":
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()

    monkeypatch.setattr(main, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(main, "reset_ingestion_state", fake_reset_ingestion_state)

    result = main.reset_index_endpoint()

    assert result.model_dump() == {"message": "Index reset successfully"}
    assert captured["path"] == raw_dir
    assert sorted(p.name for p in raw_dir.iterdir()) == [".gitkeep"]
