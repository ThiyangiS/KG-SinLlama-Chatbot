"""Normalized multilingual E5 embeddings for questions and passages."""

from .config import E5_MODEL


class E5Embeddings:
    def __init__(self, model_id=E5_MODEL, *, device=None, model=None):
        self.model_id = model_id
        self.device = device
        self._model = model

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, device=self.device, local_files_only=True
            )
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def encode_query(self, question):
        return self._encode("query: " + question)

    def encode_passages(self, passages):
        return self._encode(["passage: " + passage for passage in passages])
