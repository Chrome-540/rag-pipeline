"""Simple query tracing for the RAG pipeline."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACES_DIR = Path(__file__).parent.parent / "traces"


class Trace:
    """Bracket pipeline phases and record duration in ms."""

    def __init__(self, query: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.query = query
        self._timers: dict[str, float] = {}
        self.latency_ms: dict[str, float] = {}
        self.chunks: list[dict] = []

    def start(self, name: str):
        self._timers[name] = time.perf_counter()

    def end(self, name: str):
        if name in self._timers:
            elapsed = (time.perf_counter() - self._timers[name]) * 1000
            self.latency_ms[name] = round(elapsed, 1)

    def set_chunks(self, chunks: list[dict]):
        self.chunks = [
            {
                "page": c.get("page"),
                "score": round(c.get("score", 0), 4),
                "retrieval": c.get("retrieval", "unknown"),
                "text_preview": c.get("text", "")[:120],
            }
            for c in chunks
        ]

    def save(self) -> dict:
        total = sum(self.latency_ms.values())
        data = {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "latency_ms": self.latency_ms,
            "total_ms": round(total, 1),
            "num_chunks": len(self.chunks),
            "chunks": self.chunks,
        }
        TRACES_DIR.mkdir(exist_ok=True)
        path = TRACES_DIR / f"{self.trace_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data
