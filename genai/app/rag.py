"""
RAG pipeline: PDF → chunks → embeddings → FAISS / numpy vector search.

On first run the inverter manual is parsed, chunked, embedded, and cached
to disk so subsequent starts are instant.
"""

import os
import pickle
import numpy as np
from pathlib import Path

import fitz  # PyMuPDF

from app.config import (
    INVERTER_MANUAL_PATH,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)


class RAGPipeline:
    def __init__(self):
        self.model = None
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.ready = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def initialize(self):
        """Load cached index or build from PDF."""
        from sentence_transformers import SentenceTransformer

        print(f"[RAG] Loading embedding model '{EMBEDDING_MODEL}' …")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        store = Path(VECTOR_STORE_DIR)
        chunks_path = store / "chunks.pkl"
        emb_path = store / "embeddings.npy"

        if chunks_path.exists() and emb_path.exists():
            print("[RAG] Loading cached vector store …")
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.embeddings = np.load(str(emb_path))
            print(f"[RAG] Loaded {len(self.chunks)} chunks from cache.")
        else:
            print("[RAG] Building vector store from PDF (first run) …")
            self._build_index()

        self.ready = True
        print(f"[RAG] Ready – {len(self.chunks)} chunks indexed.")

    # ------------------------------------------------------------------
    # PDF → text → chunks → embeddings
    # ------------------------------------------------------------------
    def _extract_text_from_pdf(self) -> str:
        if not os.path.exists(INVERTER_MANUAL_PATH):
            print(f"[RAG] WARNING: PDF not found at {INVERTER_MANUAL_PATH}")
            return ""
        doc = fitz.open(INVERTER_MANUAL_PATH)
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        return chunks

    def _build_index(self):
        text = self._extract_text_from_pdf()
        if not text:
            self.chunks = []
            self.embeddings = np.array([])
            return

        self.chunks = self._chunk_text(text)
        print(f"[RAG] Encoding {len(self.chunks)} chunks …")
        self.embeddings = self.model.encode(
            self.chunks, show_progress_bar=True, batch_size=64
        )

        # Persist to disk
        store = Path(VECTOR_STORE_DIR)
        store.mkdir(parents=True, exist_ok=True)
        with open(store / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(str(store / "embeddings.npy"), self.embeddings)
        print("[RAG] Vector store cached to disk.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k most relevant chunks for *query*."""
        if not self.ready or len(self.chunks) == 0:
            return []

        k = min(top_k or TOP_K, len(self.chunks))
        q_emb = self.model.encode([query])

        # Cosine similarity via numpy
        dot = np.dot(self.embeddings, q_emb.T).flatten()
        norms = (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        sims = dot / norms

        top_idx = np.argsort(sims)[-k:][::-1]
        return [
            {"text": self.chunks[i], "score": float(sims[i])} for i in top_idx
        ]
