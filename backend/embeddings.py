"""Local embeddings via Hugging Face sentence-transformers -- no embeddings
API call, no extra API key. The model (~90MB) downloads once from the
Hugging Face Hub and is cached locally after."""

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy import: only when first used

        _model = SentenceTransformer(MODEL_ID)
    return _model


def embed(texts: list[str]):
    """Returns an (n, 384) array of L2-normalized vectors, so a dot product
    between two of them *is* their cosine similarity."""
    return _get_model().encode(texts, normalize_embeddings=True)
