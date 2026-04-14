import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================
INDEX_DIR = "politici_index"
OUTPUT_DIR = "politici_query_output"

TH_STRONG = 0.55
TH_WEAK = 0.45
TOP_K_EXAMPLES = 5

TOPICS = {
    "economia": "economia lavoro salari inflazione imprese tasse crescita pensioni",
    "immigrazione": "immigrazione confini sbarchi integrazione accoglienza clandestini sicurezza",
    "sanita": "sanita ospedali medici infermieri pronto soccorso cure liste attesa ssn",
    "scuola": "scuola insegnanti studenti universita istruzione classi formazione educazione",
    "sicurezza": "sicurezza criminalita polizia carabinieri legalita ordine pubblico violenza",
    "ambiente": "ambiente clima energia rinnovabili inquinamento transizione ecologica sostenibilita",
    "europa": "europa unione europea bruxelles euro commissione parlamento europeo trattati",
    "giustizia": "giustizia magistrati tribunali processi sentenze carcere riforma legge",
    "lavoro": "lavoro occupazione disoccupazione sindacati contratti precari cassa integrazione"
}


# =========================================================
# UTILS
# =========================================================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate(text: str, max_len: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def classify_strength(score: float, th_strong: float, th_weak: float) -> str:
    if score > th_strong:
        return "forte"
    if score > th_weak:
        return "medio"
    return "debole"


def render_html_page(title: str, df: pd.DataFrame) -> str:
    style = """
    <style>
        body { font-family: Arial, sans-serif; margin: 24px; }
        h1 { margin-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; font-size: 14px; }
        th, td { border: 1px solid #d0d0d0; padding: 8px; text-align: left; vertical-align: top; }
        th { background: #f5f5f5; position: sticky; top: 0; }
        tr:nth-child(even) { background: #fafafa; }
    </style>
    """
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        {style}
    </head>
    <body>
        <h1>{title}</h1>
        {df.to_html(index=False, escape=False)}
    </body>
    </html>
    """


# =========================================================
# MAIN
# =========================================================
def main():
    index_dir = Path(INDEX_DIR)
    output_dir = ensure_dir(OUTPUT_DIR)

    meta_path = index_dir / "index_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Indice non trovato: {meta_path}\n"
            "Prima esegui build_index_politici.py"
        )

    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    docs_path = Path(meta["docs_path"])
    emb_path = Path(meta["emb_path"])
    model_name = meta["model_name"]

    if not docs_path.exists():
        raise FileNotFoundError(f"File documenti non trovato: {docs_path}")

    if not emb_path.exists():
        raise FileNotFoundError(f"File embedding non trovato: {emb_path}")

    print("Carico indice salvato...")
    df_docs = pd.read_pickle(docs_path)
    doc_emb = np.load(emb_path)

    print("Carico il modello per le query...")
    model = SentenceTransformer(model_name)

    topic_names = list(TOPICS.keys())
    topic_queries = list(TOPICS.values())

    print("Creo embedding dei topic...")
    topic_emb = model.encode(
        topic_queries,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    score_matrix = cosine_similarity(doc_emb, topic_emb)

    summary_rows = []
    examples_rows = []

    for j, topic_name in enumerate(topic_names):
        scores = score_matrix[:, j]

        strong_mask = scores > TH_STRONG
        weak_mask = (scores > TH_WEAK) & (scores <= TH_STRONG)

        summary_rows.append(
            {
                "tema": topic_name,
                "forti": int(np.sum(strong_mask)),
                "medi": int(np.sum(weak_mask)),
                "max_score": round(float(np.max(scores)), 3),
                "p95_score": round(float(np.percentile(scores, 95)), 3),
                "media_score": round(float(np.mean(scores)), 3),
                "query": TOPICS[topic_name],
            }
        )

        top_idx = np.argsort(scores)[::-1][:TOP_K_EXAMPLES]
        for rank, idx in enumerate(top_idx, start=1):
            examples_rows.append(
                {
                    "tema": topic_name,
                    "rank": rank,
                    "score": round(float(scores[idx]), 3),
                    "folder_id": df_docs.loc[idx, "folder_id"],
                    "caption": df_docs.loc[idx, "caption"],
                    "estratto": truncate(df_docs.loc[idx, "content"]),
                }
            )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        by=["forti", "max_score", "medi"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df_examples = pd.DataFrame(examples_rows).sort_values(
        by=["tema", "rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_topic_idx = np.argmax(score_matrix, axis=1)
    best_scores = np.max(score_matrix, axis=1)

    assignment_rows = []
    for i in range(len(df_docs)):
        best_topic = topic_names[best_topic_idx[i]]
        best_score = float(best_scores[i])

        assignment_rows.append(
            {
                "doc_id": df_docs.loc[i, "doc_id"],
                "folder_id": df_docs.loc[i, "folder_id"],
                "caption": df_docs.loc[i, "caption"],
                "best_tema": best_topic,
                "best_score": round(best_score, 3),
                "forza": classify_strength(best_score, TH_STRONG, TH_WEAK),
                "estratto": truncate(df_docs.loc[i, "content"]),
            }
        )

    df_assignment = pd.DataFrame(assignment_rows).sort_values(
        by=["best_score", "folder_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df_scores = pd.DataFrame(score_matrix, columns=topic_names)
    df_scores.insert(0, "folder_id", df_docs["folder_id"])
    df_scores.insert(1, "caption", df_docs["caption"])

    for c in topic_names:
        df_scores[c] = df_scores[c].round(3)

    df_summary.to_csv(output_dir / "summary_temi.csv", index=False, encoding="utf-8")
    df_examples.to_csv(output_dir / "top_esempi.csv", index=False, encoding="utf-8")
    df_assignment.to_csv(output_dir / "best_topic_doc.csv", index=False, encoding="utf-8")
    df_scores.to_csv(output_dir / "score_completi.csv", index=False, encoding="utf-8")

    (output_dir / "summary_temi.html").write_text(
        render_html_page("Summary temi", df_summary),
        encoding="utf-8",
    )
    (output_dir / "top_esempi.html").write_text(
        render_html_page("Top esempi", df_examples),
        encoding="utf-8",
    )
    (output_dir / "best_topic_doc.html").write_text(
        render_html_page("Best topic per documento", df_assignment),
        encoding="utf-8",
    )
    (output_dir / "score_completi.html").write_text(
        render_html_page("Score completi", df_scores),
        encoding="utf-8",
    )

    print("\nQuery completata.")
    print(f"Output salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()