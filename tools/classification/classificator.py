import json
import os
import re
import requests
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════════════════════════

SMALL_DATASET = True        # ← False quando hai 6+ mesi di dati

USE_OLLAMA    = True        # ← False per usare solo KeyBERT+MMR senza LLM
OLLAMA_MODEL  = "gemma4"    # qualsiasi modello installato: llama3, mistral, phi3…
OLLAMA_URL    = "http://localhost:11434/api/generate"

# Parametri che si adattano alla dimensione del dataset
if SMALL_DATASET:
    MIN_TOPIC_SIZE   = 3
    MIN_SAMPLES      = 2
    UMAP_N_NEIGHBORS = 5
    CHUNK_MAX_WORDS  = 150
    CHUNK_MIN_WORDS  = 50
    CHUNK_OVERLAP    = 30
    NGRAM_RANGE      = (1, 2)
    MIN_DF           = 2
    OUTLIER_THRESH   = 0.05
    MERGE_SIM        = 0.80
    WORD_OVERLAP_THR = 0.50   # soglia merge per parole duplicate
else:
    MIN_TOPIC_SIZE   = 10
    MIN_SAMPLES      = 5
    UMAP_N_NEIGHBORS = 15
    CHUNK_MAX_WORDS  = 200
    CHUNK_MIN_WORDS  = 60
    CHUNK_OVERLAP    = 50
    NGRAM_RANGE      = (1, 3)
    MIN_DF           = 5
    OUTLIER_THRESH   = 0.15
    MERGE_SIM        = 0.70
    WORD_OVERLAP_THR = 0.45

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PULIZIA INPUT
# ═══════════════════════════════════════════════════════════════════════════════

def is_noisy_transcript(text: str, noise_threshold: float = 0.15) -> bool:
    """
    Rileva trascrizioni audio/video con pattern ripetitivi o OCR rotto.
    Questi testi contaminano il clustering con parole semanticamente vuote.
    """
    if not text:
        return True
    words = text.lower().split()
    if len(words) < 10:
        return True

    # Parole consecutive ripetute → trascrizione audio ("vieni vieni vieni stop…")
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    if repeats / len(words) > noise_threshold:
        return True

    # Token spazzatura da OCR: sequenze di 1-2 caratteri non alfabetici ("KE \ |")
    junk = sum(1 for w in words if len(w) <= 2 and not w.isalpha())
    if junk / len(words) > 0.20:
        return True

    return False


def clean_text(text: str) -> str:
    """Normalizzazione leggera: rimuove emoji, URL, caratteri di controllo."""
    # URL
    text = re.sub(r'https?://\S+', '', text)
    # Emoji e simboli Unicode non latini
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u20AC€£]', ' ', text)
    # Caratteri di controllo e simboli isolati (*, #, @, …)
    text = re.sub(r'(?<!\w)[*#@►▶🔵✅⚠️•·–—]+(?!\w)', ' ', text)
    # Spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_documents(json_path: str) -> tuple[list[str], list[int]]:
    """
    Carica i post dal JSON.
    Strategia: usa la caption (testo scritto) come fonte principale.
    Il campo 'text' (trascrizione audio/OCR) viene aggiunto solo se non rumoroso.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_docs, doc_ids = [], []
    skipped_noisy = 0
    skipped_short = 0

    for i, item in enumerate(data):
        caption = clean_text(item.get("caption") or "")
        text    = item.get("text") or ""

        if is_noisy_transcript(text):
            combined = caption
            skipped_noisy += 1
        else:
            combined = clean_text(f"{caption}\n{text}").strip()

        # Scarta post troppo corti (es. solo emoji o "Sì.")
        if len(combined.split()) < 15:
            skipped_short += 1
            continue

        raw_docs.append(combined)
        doc_ids.append(i)

    print(f"  Post caricati:        {len(raw_docs)}")
    print(f"  Trascrizioni scartate:{skipped_noisy}")
    print(f"  Post troppo corti:    {skipped_short}")
    return raw_docs, doc_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_words: int, overlap: int, min_words: int) -> list[str]:
    """
    Divide testi lunghi in chunk sovrapposti.
    Testi già brevi vengono restituiti interi se superano min_words.
    """
    words = text.split()

    if len(words) < min_words:
        return [text] if len(words) >= 15 else []

    if len(words) <= max_words:
        return [text]

    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)

    return chunks if chunks else [text]


def build_corpus(raw_docs: list[str], doc_ids: list[int]) -> tuple[list[str], list[int]]:
    docs, chunk_to_doc = [], []
    for original_idx, (doc, doc_id) in enumerate(zip(raw_docs, doc_ids)):
        chunks = chunk_text(
            doc,
            max_words=CHUNK_MAX_WORDS,
            overlap=CHUNK_OVERLAP,
            min_words=CHUNK_MIN_WORDS,
        )
        docs.extend(chunks)
        chunk_to_doc.extend([doc_id] * len(chunks))

    print(f"  Chunk totali: {len(docs)}")
    return docs, chunk_to_doc


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — NAMING CON OLLAMA
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_PROMPT = """Sei un analista politico italiano esperto di comunicazione social.
Ti fornisco le parole chiave e alcuni post di esempio di un cluster tematico estratto
dai contenuti social di Ilaria Salis.

Parole chiave del cluster:
{keywords}

Post di esempio:
{documents}

Restituisci ESCLUSIVAMENTE un nome breve (3-6 parole) in italiano che descriva
l'ARGOMENTO POLITICO principale del cluster. Nessuna spiegazione, nessun punto,
nessuna virgolette. Solo il nome dell'argomento."""


def name_topic_with_ollama(keywords: list[str], docs: list[str], model: str) -> str | None:
    """
    Chiama Ollama in locale per generare un nome leggibile per il topic.
    Ritorna None se Ollama non è raggiungibile (fallback a KeyBERT).
    """
    prompt = OLLAMA_PROMPT.format(
        keywords=", ".join(keywords[:10]),
        documents="\n---\n".join(d[:300] for d in docs[:3])
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        resp.raise_for_status()
        name = resp.json().get("response", "").strip()
        # Sanity check: scarta risposte troppo lunghe o vuote
        if 2 <= len(name.split()) <= 10:
            return name
        # Se il modello ha risposto con più righe, prendi la prima
        first_line = name.split("\n")[0].strip()
        if 2 <= len(first_line.split()) <= 10:
            return first_line
        return None
    except requests.exceptions.ConnectionError:
        print("  ⚠ Ollama non raggiungibile — uso nomi KeyBERT")
        return None
    except Exception as e:
        print(f"  ⚠ Errore Ollama ({e}) — uso nomi KeyBERT")
        return None


def enrich_topic_names(topic_model: BERTopic, docs: list[str]) -> dict[int, str]:
    """
    Per ogni topic, genera un nome leggibile via Ollama.
    Ritorna un dizionario {topic_id: "Nome argomento"}.
    """
    topic_info = topic_model.get_topic_info()
    names = {}

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            names[topic_id] = "Outlier / Non classificato"
            continue

        keywords = [w for w, _ in topic_model.get_topic(topic_id)]

        # Recupera i doc rappresentativi dal modello
        repr_docs = topic_model.representative_docs_.get(topic_id, [])
        if not repr_docs:
            # Fallback: prendi i primi doc assegnati a questo topic
            repr_docs = [d for d, t in zip(docs, topic_model.topics_) if t == topic_id][:3]

        name = name_topic_with_ollama(keywords, repr_docs, OLLAMA_MODEL)
        if name:
            names[topic_id] = name
            print(f"  Topic {topic_id:2d}: {name}")
        else:
            # Fallback: costruisci nome dalle top-2 parole chiave
            names[topic_id] = " / ".join(keywords[:2]).title()
            print(f"  Topic {topic_id:2d}: {names[topic_id]}  (fallback KeyBERT)")

    return names


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — POST-PROCESSING: MERGE DUPLICATI
# ═══════════════════════════════════════════════════════════════════════════════

def find_duplicate_topics(
    topic_model: BERTopic,
    word_overlap_thr: float = 0.50,
    embed_sim_thr: float = 0.80,
) -> list[list[int]]:
    """
    Identifica topic da unire usando due criteri combinati:
    1. Overlap nelle top-words (cattura duplicati lessicali come topic 2 e 7)
    2. Cosine similarity tra embedding dei centroidi (cattura duplicati semantici)
    """
    topic_ids = [t for t in topic_model.get_topics() if t != -1]
    topics_words = {t: set(w for w, _ in topic_model.get_topic(t)) for t in topic_ids}

    pairs_to_merge = set()

    # Criterio 1: overlap parole
    for i, ti in enumerate(topic_ids):
        for tj in topic_ids[i + 1:]:
            a, b = topics_words[ti], topics_words[tj]
            if not a or not b:
                continue
            overlap = len(a & b) / min(len(a), len(b))
            if overlap >= word_overlap_thr:
                pairs_to_merge.add((min(ti, tj), max(ti, tj)))

    # Criterio 2: similarity embedding
    embeddings = topic_model.topic_embeddings_
    if embeddings is not None and len(embeddings) > 2:
        # topic_embeddings_ ha shape (n_topics+1, dim), il primo è -1
        valid = [(t, t + 1) for t in topic_ids if t + 1 < len(embeddings)]
        for i, (ti, ei) in enumerate([(t, t + 1) for t in topic_ids]):
            for tj, ej in [(t, t + 1) for t in topic_ids[i + 1:]]:
                if ei < len(embeddings) and ej < len(embeddings):
                    sim = cosine_similarity(
                        embeddings[ei].reshape(1, -1),
                        embeddings[ej].reshape(1, -1)
                    )[0][0]
                    if sim >= embed_sim_thr:
                        pairs_to_merge.add((min(ti, tj), max(ti, tj)))

    # Raggruppa le coppie in cluster connessi (union-find semplice)
    if not pairs_to_merge:
        return []

    parent = {t: t for t in topic_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    for a, b in pairs_to_merge:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for t in topic_ids:
        root = find(t)
        groups.setdefault(root, []).append(t)

    return [g for g in groups.values() if len(g) > 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — PIPELINE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path   = os.path.join(base_dir, 'data', 'content', 'matteosalviniofficial', 'matteosalviniofficial.json')
    output_dir  = os.path.dirname(__file__)

    mode_label = "SMALL (1 mese)" if SMALL_DATASET else "LARGE (6+ mesi)"
    print(f"\n{'═'*60}")
    print(f"  BERTopic Salvini  —  modalità {mode_label}")
    print(f"{'═'*60}\n")

    # ── 1. Caricamento e pulizia ─────────────────────────────────────────────
    print("[ 1/6 ] Caricamento e pulizia documenti...")
    raw_docs, doc_ids = load_documents(json_path)

    # ── 2. Chunking ──────────────────────────────────────────────────────────
    print("\n[ 2/6 ] Chunking...")
    docs, chunk_to_doc = build_corpus(raw_docs, doc_ids)

    if len(docs) < 10:
        print("  ✗ Troppo pochi chunk. Controlla il dataset o abbassa CHUNK_MIN_WORDS.")
        return

    # ── 3. Stopwords e vectorizer ────────────────────────────────────────────
    print("\n[ 3/6 ] Configurazione modello...")
    nltk.download('stopwords', quiet=True)
    italian_sw = stopwords.words('italian')
    extra_sw = [
        # generici
        "essere", "avere", "fare", "così", "quindi", "quando", "quello",
        "tutto", "tutti", "solo", "questo", "questa", "questi", "più",
        "cui", "perché", "già", "dopo", "oggi", "ieri", "domani", "poi",
        "qui", "lì", "ci", "ne", "anche", "però", "invece", "allora",
        "sempre", "ancora", "molto", "bene", "dire", "andare", "vedere",
        # social/filler
        "grazie", "amici", "ciao", "buona", "giornata", "buon", "buongiorno",
        "sì", "no", "ecco", "anzi", "adesso", "ora", "proprio", "vieni",
        # troppo generici nel contesto politico
        "anni", "anno", "volta", "cosa", "come", "hanno", "sono", "stato",
        "tanti", "fatto", "avanti", "qua", "finalmente", "tornare",
        "soprattutto", "territorio", "stata", "fino",
    ]
    italian_sw.extend(extra_sw)

    vectorizer = CountVectorizer(
        stop_words=italian_sw,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
    )

    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.4),
    ]

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=True,
        verbose=False,
    )

    # ── 4. Fit + riduzione outlier ───────────────────────────────────────────
    print("\n[ 4/6 ] Training BERTopic...")
    topics, probs = topic_model.fit_transform(docs)

    n_outliers_before = sum(1 for t in topics if t == -1)
    topics = topic_model.reduce_outliers(docs, topics, threshold=OUTLIER_THRESH)
    topic_model.update_topics(
        docs, topics=topics,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
    )
    n_outliers_after = sum(1 for t in topics if t == -1)
    n_topics_raw = len([t for t in topic_model.get_topic_info()["Topic"] if t != -1])
    print(f"  Topic trovati:    {n_topics_raw}")
    print(f"  Outlier: {n_outliers_before} → {n_outliers_after}")

    # ── 5. Merge topic duplicati ─────────────────────────────────────────────
    print("\n[ 5/6 ] Ricerca e merge topic duplicati...")
    dup_groups = find_duplicate_topics(
        topic_model,
        word_overlap_thr=WORD_OVERLAP_THR,
        embed_sim_thr=MERGE_SIM,
    )
    if dup_groups:
        for group in dup_groups:
            words_preview = {t: [w for w, _ in topic_model.get_topic(t)][:3] for t in group}
            print(f"  Merge {group}: {words_preview}")
        topic_model.merge_topics(docs, dup_groups)
        n_topics_final = len([t for t in topic_model.get_topic_info()["Topic"] if t != -1])
        print(f"  Topic dopo merge: {n_topics_final}")
    else:
        print("  Nessun duplicato trovato.")

    # ── 6. Naming con Ollama ─────────────────────────────────────────────────
    topic_names = {}
    if USE_OLLAMA:
        print(f"\n[ 6/6 ] Naming argomenti con Ollama ({OLLAMA_MODEL})...")
        topic_names = enrich_topic_names(topic_model, docs)
    else:
        print("\n[ 6/6 ] Naming saltato (USE_OLLAMA=False), uso rappresentazione KeyBERT.")

    # ── Export ───────────────────────────────────────────────────────────────
    print("\n  Esportazione CSV...")
    df_info = topic_model.get_topic_info()

    # Aggiungi colonna con il nome Ollama se disponibile
    if topic_names:
        df_info["OllamaName"] = df_info["Topic"].map(topic_names)

    df_info.to_csv(os.path.join(output_dir, 'topics_summary.csv'), index=False)

    # CSV chunk-level
    df_docs = pd.DataFrame({
        "Chunk":  docs,
        "Doc_ID": chunk_to_doc,
        "Topic":  topics,
    }).merge(df_info[["Topic", "Name"] + (["OllamaName"] if topic_names else [])],
             on="Topic", how="left") \
      .sort_values("Topic")
    df_docs.to_csv(os.path.join(output_dir, 'documents_with_topics.csv'), index=False)

    # CSV doc-level (un post = una riga, con tutti i topic che lo coprono)
    name_col = "OllamaName" if topic_names else "Name"
    df_agg = df_docs.groupby("Doc_ID").agg(
        Topics=(    "Topic",    lambda x: list(x.unique())),
        TopicNames=(name_col,   lambda x: list(x.dropna().unique())),
        Chunk=(     "Chunk",    "first"),
    ).reset_index()
    df_agg.to_csv(os.path.join(output_dir, 'docs_aggregated.csv'), index=False)

    # ── Visualizzazioni ──────────────────────────────────────────────────────
    print("  Visualizzazioni...")
    for name, fn in [
        ("barchart",  topic_model.visualize_barchart),
        ("heatmap",   topic_model.visualize_heatmap),
        ("hierarchy", topic_model.visualize_hierarchy),
    ]:
        try:
            fn().write_html(os.path.join(output_dir, f'topics_{name}.html'))
        except Exception as e:
            print(f"  Saltata visualizzazione '{name}': {e}")

    # ── Salva modello ────────────────────────────────────────────────────────
    topic_model.save(
        os.path.join(output_dir, 'salvini_bertopic_model'),
        serialization="safetensors",
        save_ctfidf=True,
    )

    # ── Summary finale ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  RISULTATI FINALI")
    print(f"{'═'*60}")
    final_info = topic_model.get_topic_info()
    for _, row in final_info.iterrows():
        tid  = row["Topic"]
        cnt  = row["Count"]
        name = topic_names.get(tid, row["Name"]) if topic_names else row["Name"]
        marker = "⚠" if tid == -1 else " "
        print(f"  {marker} Topic {tid:3d}  ({cnt:3d} chunk)  {name}")
    print(f"\n  Output in: {output_dir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()