"""Seed script — downloads the dataset and caches it on disk.

Called during ``docker build`` so that containers start instantly
without fetching from Hugging Face at runtime.

Usage:
    python -m scripts.download_and_cache [--output-dir /path/to/cache]
"""

from __future__ import annotations

import argparse
import io
import pathlib

import numpy as np
from datasets import load_dataset


def download_and_cache(output_dir: pathlib.Path) -> None:
    """Download ``jacob-ml/LS-Images`` and write cache files.

    Writes two files:
        embeddings.npz   –  (443, 256) float32 array, numpy-native
        rows.pkl         –  list of (id: str, image_bytes: bytes) tuples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset …")
    ds = load_dataset("jacob-ml/LS-Images", split="train")

    ids = list(ds["id"])
    embeddings_raw = list(ds["embedding"])
    images = ds["image"]

    # Stack embeddings into a single contiguous float32 array.
    embeddings = np.stack(
        [np.array(e, dtype=np.float32) for e in embeddings_raw]
    )
    assert embeddings.shape == (443, 256), f"Unexpected shape: {embeddings.shape}"

    # Serialize images as raw PNG bytes.
    rows = []
    for idx in range(len(ids)):
        buf = io.BytesIO()
        images[idx].save(buf, format="PNG")
        rows.append((ids[idx], buf.getvalue()))

    np.savez_compressed(output_dir / "embeddings.npz", data=embeddings)
    import pickle

    with open(output_dir / "rows.pkl", "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cache written to {output_dir}")
    print(f"  embeddings.npz: {embeddings.nbytes:,} bytes (compressed)")
    print(f"  rows.pkl:       {pathlib.Path(output_dir / 'rows.pkl').stat().st_size:,} bytes")
    print(f"  items:          {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("cache"),
        help="Directory to write cache files (default: ./cache)",
    )
    args = parser.parse_args()

    download_and_cache(args.output_dir)


if __name__ == "__main__":
    main()
