"""Run an evidence-grounded extraction with a resumable per-chunk cache."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from .client import ChatClient
from .experiment import EXPERIMENT_PROMPT, EXPERIMENT_SCHEMA, normalize_experiments
from .knowledge import KNOWLEDGE_PROMPT, normalize_knowledge
from .pdf import PDFExtractionError, extract_pdf

PROMPTS = {
    "knowledge": (KNOWLEDGE_PROMPT, normalize_knowledge),
    "experiment": (EXPERIMENT_PROMPT + "\nJSON schema:\n" + json.dumps(EXPERIMENT_SCHEMA), normalize_experiments),
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace a completed artifact without leaving a partially written file."""
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pdfs(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".pdf":
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    return []


def _ground(memories: list[dict], doc: Any, chunk: Any, model: str) -> list[dict]:
    grounded = []
    for memory in memories:
        quote = memory["evidence"]
        start = chunk.text.find(quote)
        if start < 0:
            raise ValueError("Model evidence is not a verbatim substring of the source chunk")
        page_start = chunk.start_offset + start
        identity = json.dumps([doc.document_id, chunk.page_start, page_start, quote,
                               memory["memory_type"], memory["subtype"], memory["statement"]], ensure_ascii=False)
        grounded.append({
            "schema_version": "1.0", "memory_id": _hash(identity),
            "memory_type": memory["memory_type"], "subtype": memory["subtype"],
            "statement": memory["statement"], "details": memory["details"],
            "source": {"document_id": doc.document_id, "file_name": Path(doc.source_path).name,
                       "page": chunk.page_start, "chunk_id": chunk.chunk_id,
                       "page_char_start": page_start, "page_char_end": page_start + len(quote),
                       "quote": quote},
            "extraction": {"model": model},
        })
    return grounded


def run(source: str | Path, output_dir: str | Path = "memory-output", *,
        model: str = "gpt-4o-mini", types: tuple[str, ...] = ("knowledge", "experiment"),
        client: Any = None, chunk_size: int = 10000, chunk_overlap: int = 1000,
        parse_only: bool = False, on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Extract from a PDF or recursively from a folder.

    A custom client must implement ``complete(model=, system=, text=) -> dict``.
    Successful chunk/type calls are cached in output_dir; failures are retried on
    the next run. Each invocation writes a separate immutable result snapshot.
    """
    selected = tuple(dict.fromkeys(types))
    if not selected or any(t not in PROMPTS for t in selected):
        raise ValueError("types must contain knowledge and/or experiment")
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= chunk_overlap < chunk_size")
    pdfs = _pdfs(Path(source))
    if not pdfs:
        raise ValueError(f"No PDF files found in {source}")
    if not parse_only and client is None:
        client = ChatClient()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    snapshot = output / "runs" / run_id
    snapshot.mkdir(parents=True)
    results: dict[str, list[dict]] = {t: [] for t in PROMPTS}
    documents, chunks, errors = [], [], []
    seen_docs, seen_memories = set(), set()
    calls = cached = 0
    connection = sqlite3.connect(output / "cache.sqlite3")
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS results (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for path in pdfs:
            try:
                doc = extract_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except (PDFExtractionError, OSError) as exc:
                errors.append({"file_name": path.name, "stage": "pdf", "error": str(exc)})
                continue
            if doc.document_id in seen_docs:
                continue
            seen_docs.add(doc.document_id)
            documents.append({"document_id": doc.document_id, "file_name": path.name,
                              "metadata": doc.metadata, "empty_pages": doc.empty_pages,
                              "chunk_count": len(doc.chunks)})
            for chunk in doc.chunks:
                chunks.append({"document_id": doc.document_id, **asdict(chunk)})
                if parse_only:
                    continue
                for kind in selected:
                    prompt, normalize = PROMPTS[kind]
                    endpoint = getattr(client, "base_url", type(client).__module__ + "." + type(client).__qualname__)
                    key = _hash(json.dumps(["v1", doc.document_id, chunk.chunk_id, kind, model, endpoint, prompt], ensure_ascii=False))
                    try:
                        row = connection.execute("SELECT payload FROM results WHERE cache_key = ?", (key,)).fetchone()
                        if row:
                            try:
                                memories = _ground(normalize(json.loads(row[0])), doc, chunk, model)
                            except (ValueError, KeyError, TypeError):
                                connection.execute("DELETE FROM results WHERE cache_key = ?", (key,))
                                connection.commit()
                                row = None
                            else:
                                cached += 1
                        if not row:
                            calls += 1
                            payload = client.complete(model=model, system=prompt, text=chunk.text)
                            memories = _ground(normalize(payload), doc, chunk, model)
                            connection.execute("INSERT OR REPLACE INTO results VALUES (?, ?)", (key, json.dumps(payload, ensure_ascii=False)))
                            connection.commit()
                        for memory in memories:
                            if memory["memory_id"] not in seen_memories:
                                seen_memories.add(memory["memory_id"])
                                results[kind].append(memory)
                    except Exception as exc:
                        errors.append({"file_name": path.name, "document_id": doc.document_id,
                                       "chunk_id": chunk.chunk_id, "page": chunk.page_start,
                                       "stage": kind, "error": f"{type(exc).__name__}: {exc}"})
                        task_status = "error"
                    else:
                        task_status = "cached" if row else "complete"
                    if on_progress is not None:
                        on_progress({"file_name": path.name, "page": chunk.page_start,
                                     "chunk_id": chunk.chunk_id, "type": kind, "status": task_status})
    finally:
        connection.close()
    if not parse_only:
        for kind in selected:
            _write_jsonl(snapshot / f"{kind}.jsonl", results[kind])
    _write_jsonl(snapshot / "documents.jsonl", documents)
    _write_jsonl(snapshot / "chunks.jsonl", chunks)
    _write_jsonl(snapshot / "errors.jsonl", errors)
    summary = {"status": "partial" if errors else ("parsed" if parse_only else "complete"),
               "input_pdf_count": len(pdfs), "documents": len(documents), "chunks": len(chunks),
               "knowledge": len(results["knowledge"]), "experiment": len(results["experiment"]),
               "errors": len(errors), "model_calls": calls, "cached_calls": cached,
               "model": None if parse_only else model, "types": list(selected), "run_dir": str(snapshot)}
    _write_jsonl(snapshot / "manifest.jsonl", [summary])
    return summary


def iter_memories(run_dir: str | Path, memory_type: str) -> Iterator[dict[str, Any]]:
    """Stream one selected memory type from a run snapshot."""
    if memory_type not in PROMPTS:
        raise ValueError("memory_type must be knowledge or experiment")
    with (Path(run_dir) / f"{memory_type}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
