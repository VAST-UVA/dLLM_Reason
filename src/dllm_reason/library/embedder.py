"""Task embedding: encode task descriptions into dense vectors for retrieval.

Abstract interface + concrete implementations. Easily extensible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class TaskEmbedder(ABC):
    """Encode a task description string into a fixed-dim dense vector."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return shape (dim,) float32 vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return shape (N, dim) float32 matrix."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class SentenceTransformerEmbedder(TaskEmbedder):
    """Wraps sentence-transformers for task embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)


class RandomEmbedder(TaskEmbedder):
    """Deterministic random embedder for testing / ablation (no model needed)."""

    def __init__(self, dim: int = 384, seed: int = 42):
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class TFIDFEmbedder(TaskEmbedder):
    """Lightweight TF-IDF + SVD embedder (no neural model)."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._dim = dim
        self._tfidf = TfidfVectorizer(max_features=5000)
        self._svd = TruncatedSVD(n_components=dim)
        self._fitted = False
        self._corpus: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        self._corpus = corpus
        X = self._tfidf.fit_transform(corpus)
        n_components = min(self._dim, X.shape[1], X.shape[0])
        self._svd = type(self._svd)(n_components=n_components)
        self._svd.fit(X)
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder.fit() must be called before embed().")
        X = self._tfidf.transform([text])
        vec = self._svd.transform(X)[0].astype(np.float32)
        # Pad if SVD components < dim
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


def create_embedder(name: str, **kwargs) -> TaskEmbedder:
    """Factory for embedders."""
    if name == "sentence_transformer":
        return SentenceTransformerEmbedder(**kwargs)
    elif name == "random":
        return RandomEmbedder(**kwargs)
    elif name == "tfidf":
        return TFIDFEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder: {name}")
