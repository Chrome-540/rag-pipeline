import os
import json
from src.logger import get_logger

log = get_logger("parent_store")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
PARENT_FILE = os.path.join(CACHE_DIR, "parent_chunks.json")

_parent_cache = None


def save_parents(parents: dict):
    """Save parent chunks to local cache."""
    global _parent_cache
    log.info(f">> save_parents | input: {len(parents)} parents")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(PARENT_FILE, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    _parent_cache = parents
    log.info(f"<< save_parents | output: saved to {PARENT_FILE}")


def load_parents() -> dict:
    """Load parent chunks from cache."""
    global _parent_cache
    if _parent_cache is not None:
        return _parent_cache

    if not os.path.exists(PARENT_FILE):
        log.info(f"<< load_parents | no cache found")
        return {}

    with open(PARENT_FILE, "r", encoding="utf-8") as f:
        _parent_cache = json.load(f)
    log.info(f"<< load_parents | loaded {len(_parent_cache)} parents")
    return _parent_cache


def get_parent_text(parent_id: str) -> str:
    """Look up parent text by ID."""
    parents = load_parents()
    parent = parents.get(parent_id)
    if parent:
        return parent["text"]
    return ""