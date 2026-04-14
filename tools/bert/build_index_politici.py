import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================
INPUT_JSON = "ellyesse.json"
OUTPUT_DIR = "politici_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# =========================================================
# UTILS
# =========================================================
def load_json(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Il JSON deve contenere una lista di record.")

    return data


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def join_caption_text(record: dict) -> str:
    caption = str(record.get("caption", "")).strip()
    text = str(record.get("text", "")).strip()

    if caption and text:
        return f"{caption} {text}"
    return caption or text


def build_documents_dataframe(data: list[dict]) -> pd.DataFrame:
    rows = []

    for i, item in enumerate(data):
        rows.append(
            {
                "doc_id": i,
                "folder_id": item.get("folder_id", ""),
                "type": item.get("type", ""),
                "language": item.get("language", ""),
                "caption": str(item.get("caption", "")).strip(),
                "text": str(item.get("text", "")).strip(),
                "content": join_caption_text(item),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Dataset vuoto.")

    if (df["content"].fillna("").str.len() == 0).all():
        raise ValueError("Tutti i documenti hanno content vuoto.")

    return df


# =========================================================
# MAIN
# =========================================================
def main():
    output_dir = ensure_dir(OUTPUT_DIR)
    data = load_json(INPUT_JSON)
    df_docs = build_documents_dataframe(data)

    print("Carico il modello...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creo gli embedding dei documenti...")
    doc_embeddings = model.encode(
        df_docs["content"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    docs_path = output_dir / "documents.pkl"
    emb_path = output_dir / "doc_embeddings.npy"
    meta_path = output_dir / "index_meta.pkl"

    df_docs.to_pickle(docs_path)
    np.save(emb_path, doc_embeddings)

    meta = {
        "model_name": MODEL_NAME,
        "n_docs": len(df_docs),
        "docs_path": str(docs_path),
        "emb_path": str(emb_path),
        "content_definition": "caption + text",
    }

    with meta_path.open("wb") as f:
        pickle.dump(meta, f)

    print("\nIndice creato correttamente.")
    print(f"Documenti salvati in: {docs_path}")
    print(f"Embedding salvati in: {emb_path}")
    print(f"Metadati salvati in: {meta_path}")


if __name__ == "__main__":
    main()