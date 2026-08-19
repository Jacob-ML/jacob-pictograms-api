"""In-memory store for dataset embeddings and images.

Supports two loading modes:
1. **Live** – fetches ``jacob-ml/LS-Images`` from Hugging Face Hub at runtime.
2. **Cached** – loads pre-baked files written by
   ``scripts/download_and_cache.py`` (used in Docker builds).

In both cases the store is fully populated on construction so that every
request can be served without I/O latency.
"""

from __future__ import annotations

import base64
import io
import os
import pickle
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "jacob-ml/LS-Images"
CACHE_DIR = os.environ.get("PICTOGRAM_CACHE_DIR", "/app/cache")

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    """One row from the dataset, stored in query-friendly shapes."""

    id: str
    embedding: np.ndarray  # shape (256,), dtype float32
    image_bytes: bytes     # raw PNG bytes


class EmbeddingStore:
    """Loads and serves embeddings + images for similarity search.

    Parameters
    ----------
    dataset_name :
        Only used when loading live from Hugging Face.  Ignored in cached mode.
    cache_dir :
        Directory containing ``embeddings.npz`` and ``rows.pkl``.
        Set the ``PICTOGRAM_CACHE_DIR`` env var to override the default.
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        cache_dir: str | None = None,
    ) -> None:
        self._rows: list[_Row] = []
        self._embeddings: np.ndarray | None = None
        cache_dir = cache_dir or CACHE_DIR
        if os.path.isdir(cache_dir) and (
            os.path.isfile(os.path.join(cache_dir, "embeddings.npz"))
            and os.path.isfile(os.path.join(cache_dir, "rows.pkl"))
        ):
            self._load_cached(cache_dir)
        else:
            self._load_live(dataset_name)

    # -- live loading (Hugging Face Hub) -------------------------------------

    def _load_live(self, dataset_name: str) -> None:
        ds = load_dataset(dataset_name, split="train")

        images = ds["image"]
        ids = list(ds["id"])
        embeddings_raw = list(ds["embedding"])

        self._rows = []
        for idx in range(len(ids)):
            buf = io.BytesIO()
            images[idx].save(buf, format="PNG")
            self._rows.append(
                _Row(
                    id=ids[idx],
                    embedding=np.array(embeddings_raw[idx], dtype=np.float32),
                    image_bytes=buf.getvalue(),
                )
            )

        # Stack into a single 2-D array for vectorised dot-product search.
        self._embeddings = np.stack([r.embedding for r in self._rows])

    # -- cached loading (pre-baked from docker build) ------------------------

    def _load_cached(self, cache_dir: str) -> None:
        path = os.path.join(cache_dir, "embeddings.npz")
        self._embeddings = np.load(path)["data"].astype(np.float32)

        with open(os.path.join(cache_dir, "rows.pkl"), "rb") as f:
            rows_raw: list[tuple[str, bytes]] = pickle.load(f)

        # Reconstruct _rows with embeddings from the stacked array.
        self._rows = [
            _Row(id=id_, embedding=self._embeddings[i], image_bytes=bytes_)
            for i, (id_, bytes_) in enumerate(rows_raw)
        ]

    # -- query interface -----------------------------------------------------

    def search(self, query_embedding: np.ndarray) -> int:
        """Return the index of the row with highest dot-product score.

        Parameters
        ----------
        query_embedding:
            1-D array of shape ``(dim,)``, same dtype as stored embeddings.

        Returns
        -------
        int
            Index into ``self._rows`` / ``self._embeddings``.
        """
        if self._embeddings is None:
            raise RuntimeError("EmbeddingStore has not been loaded.")

        scores = self._embeddings @ query_embedding  # (N,)  -- vectorised
        return int(np.argmax(scores))

    def get_result(self, index: int) -> dict:
        """Return the base64-encoded image and metadata for *index*."""
        row = self._rows[index]
        return {
            "id": row.id,
            "image_base64": base64.b64encode(row.image_bytes).decode("ascii"),
            "content_type": "image/png",
        }

    def count(self) -> int:
        """Number of items in the store."""
        return len(self._rows)
