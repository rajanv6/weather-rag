import threading
from dataclasses import asdict, dataclass

import numpy as np

from data import CITIES, Snapshot, load_snapshot
from embeddings import embed
from places import detect_cities

_snapshot: Snapshot | None = None
_vectors: np.ndarray | None = None
_lock = threading.Lock()

KIND_ORDER = {"forecast": 0, "past_week": 1, "year_ago": 2}
MAX_CITIES = 3


def build_index(force: bool = False) -> None:
    """Fetches the weather (or reads the <1h-old disk cache) and embeds every
    note. Also the hourly refresh: the new snapshot and vectors are swapped in
    together, so a request never sees notes and vectors from different fetches."""
    global _snapshot, _vectors
    snap = load_snapshot(force)
    vectors = embed([d.text for d in snap.docs])
    with _lock:
        _snapshot, _vectors = snap, vectors
    print(f"RAG index built: {len(snap.docs)} docs embedded for {len(CITIES)} cities.")


def snapshot() -> Snapshot:
    if _snapshot is None:
        raise RuntimeError("Index not built yet -- call build_index() first.")
    return _snapshot


@dataclass
class RetrievedDoc:
    city: str
    text: str
    score: float
    kind: str


def retrieve(query: str) -> list[RetrievedDoc]:
    """Named city -> all of its notes (forecast, past week, year ago), ordered by
    mention. No named city -> nothing: a few "closest" notes out of ~100 cities
    mislead the model into ranking from a sample, so those questions (regions,
    rankings, "worldwide") are answered by tools.rank_cities instead."""
    with _lock:
        snap, vectors = _snapshot, _vectors
    if snap is None or vectors is None:
        raise RuntimeError("Index not built yet -- call build_index() first.")
    docs = snap.docs
    cities = detect_cities(query)[:MAX_CITIES]
    if not cities:
        return []
    scores = vectors @ embed([query])[0]  # cosine similarity (vectors are normalized)
    idx = [i for i, d in enumerate(docs) if d.city in cities]
    idx.sort(key=lambda i: (cities.index(docs[i].city), KIND_ORDER[docs[i].kind]))
    return [RetrievedDoc(docs[i].city, docs[i].text, float(scores[i]), docs[i].kind) for i in idx]


def to_dict(doc: RetrievedDoc) -> dict:
    return asdict(doc)
