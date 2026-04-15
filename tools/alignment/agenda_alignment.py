"""
Agenda Alignment Score
======================
Measures how well politician social content aligns with youth concerns.

Pipeline:
  1. Define concern categories (edit CATEGORIES below — no retraining needed)
  2. Build youth distribution:
       - ISTAT mode:  parse ground_truth_multiyr.json (structured survey data)
       - [OPTIONAL] Text mode: run NLI on any unstructured text corpus
  3. Run NLI zero-shot on politician chunks → politician distribution
  4. Compute JSD → Agenda Alignment Score per politician

To add/remove categories: edit CATEGORIES. Rerun. Done.
"""

import json
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from transformers import pipeline
import torch

# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORIES — edit freely, rerun to get new results
#  Each value is the NLI hypothesis fed to the model.
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    # Broadened vocabulary: captures "transizione ecologica", "green deal", ecc.
    "clima":        "Questo testo esprime preoccupazione per il cambiamento climatico, "
                    "la transizione ecologica, le energie rinnovabili o l'emergenza ambientale.",

    # Anchored to lived experience (precariato, affitti, stipendi bassi)
    "economia":     "Questo testo descrive difficoltà economiche vissute in prima persona, "
                    "come precarietà lavorativa, disoccupazione giovanile, salari bassi o "
                    "costo della vita insostenibile.",

    # Scoped to progressive civil rights
    "diritti":      "Questo testo tratta di diritti civili, pari opportunità, parità di genere, "
                    "diritti LGBTQ+ o lotta alla discriminazione.",

    # Unchanged — well-scoped already
    "sicurezza":    "Questo testo tratta di criminalità, sicurezza urbana, ordine pubblico "
                    "o paura della violenza.",

    # Key fix: experiential disillusionment
    "sfiducia":     "Questo testo esprime disincanto personale, astensionismo o sfiducia "
                    "verso i partiti politici e le istituzioni democratiche.",

    "immigrazione": "Questo testo tratta di immigrazione, politiche migratorie, "
                    "accoglienza o integrazione dei migranti.",

    # Propaganda penalizza i politici rallentando il loro allineamento
    "propaganda":   "Questo testo contiene attacchi politici contro avversari, "
                    "propaganda elettorale o polemiche di partito senza proposta concreta.",

    # Bucket per tutto ciò che non supera la soglia di confidenza (rumore di fondo)
    "altro":        "Questo testo tratta di argomenti generici, lifestyle, o argomenti non politici."
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SMALL_RUN_CHUNKS      = 30    # chunks per politician (quick test)
SMALL_RUN_ARCHETYPES  = 10    # archetypes from ground truth (quick test)

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTENT_DIR    = os.path.join(BASE_DIR, "data", "content")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# ── Modalità ISTAT (Strutturata - Default Scientifico) ──
YOUTH_SOURCE       = "istat"
GROUND_TRUTH_PATH  = os.path.join(PROCESSED_DIR, "ground_truth_multiyr.json")

# ── [FUTURO] Modalità Testo (Non Strutturata) ──
# Decommentare le righe sottostanti se si dispone di post reali di giovani
# YOUTH_SOURCE       = "text"
# YOUTH_TEXT_PATH    = os.path.join(CONTENT_DIR, "dataset_giovani_reali.json")

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Chunking
CHUNK_MAX_WORDS = 150
CHUNK_MIN_WORDS = 30
CHUNK_OVERLAP   = 30

NLI_BATCH_SIZE = 8

# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u20AC€£]', ' ', text)
    text = re.sub(r'(?<!\w)[*#@►▶•·–—]+(?!\w)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_noisy(text: str, threshold: float = 0.15) -> bool:
    if not text:
        return True
    words = text.lower().split()
    if len(words) < 10:
        return True
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    if repeats / len(words) > threshold:
        return True
    junk = sum(1 for w in words if len(w) <= 2 and not w.isalpha())
    if junk / len(words) > 0.20:
        return True
    return False

def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) < CHUNK_MIN_WORDS:
        return [text] if len(words) >= 15 else []
    if len(words) <= CHUNK_MAX_WORDS:
        return [text]
    chunks, step = [], CHUNK_MAX_WORDS - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_MAX_WORDS])
        if len(chunk.split()) >= CHUNK_MIN_WORDS:
            chunks.append(chunk)
    return chunks or [text]

def load_politician_chunks(json_path: str) -> list[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for item in data:
        caption = clean_text(item.get("caption") or "")
        text    = item.get("text") or ""
        combined = caption if is_noisy(text) else clean_text(f"{caption}\n{text}").strip()
        if len(combined.split()) >= 15:
            chunks.extend(chunk_text(combined))
    return chunks

# ═══════════════════════════════════════════════════════════════════════════════
#  YOUTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

_ISTAT_FLAGS = {
    "clima":     "fortemente preoccupato per il cambiamento climatico",
    "economia":  "insoddisfazione economica",
    "diritti":   "diritti civili",
    "sicurezza": "teme la criminalità",
    "sfiducia":  "sfiducia verso i partiti",
}

def build_istat_distribution(gt_path: str, small_run: bool = False) -> np.ndarray:
    with open(gt_path, encoding="utf-8") as f:
        archetypes = json.load(f)

    if small_run:
        archetypes = archetypes[:SMALL_RUN_ARCHETYPES]
        print(f"  [SMALL_RUN] Using {SMALL_RUN_ARCHETYPES} archetypes")

    cat_keys        = list(CATEGORIES.keys())
    meaningful_keys = cat_keys
    dist            = np.zeros(len(cat_keys))

    istat_mapped = {k: v for k, v in _ISTAT_FLAGS.items() if k in meaningful_keys}
    nli_needed   = [k for k in meaningful_keys if k not in istat_mapped]

    for archetype in archetypes:
        w    = archetype["weight_pct"] / 100.0
        desc = archetype["profile_description"].lower()
        for k, flag in istat_mapped.items():
            if flag.lower() in desc:
                dist[cat_keys.index(k)] += w

    if nli_needed:
        nli_hypotheses = [CATEGORIES[k] for k in nli_needed]
        classifier = _get_classifier()

        for archetype in archetypes:
            w          = archetype["weight_pct"] / 100.0
            statements = archetype.get("generated_statements", [])
            if not statements:
                continue
            scores = _classify_chunks(classifier, statements, nli_hypotheses)
            avg    = scores.mean(axis=0) 
            for i, k in enumerate(nli_needed):
                dist[cat_keys.index(k)] += avg[i] * w

    # Forza a 0 il rumore per il profilo target ideale
    if "propaganda" in cat_keys:
        dist[cat_keys.index("propaganda")] = 0.0
    if "altro" in cat_keys:
        dist[cat_keys.index("altro")] = 0.0

    meaningful_idx = [cat_keys.index(k) for k in meaningful_keys]
    total = dist[meaningful_idx].sum()
    dist[meaningful_idx] /= total
    return dist


# ── [OPZIONALE] BLOCCO PER TESTI NON STRUTTURATI GIOVANI ──
# Decommentare questa funzione se si passa a YOUTH_SOURCE = "text"
# def build_text_distribution(text_json_path: str, classifier) -> np.ndarray:
#     """
#     Costruisce la distribuzione dei giovani analizzando post reali (non ISTAT).
#     Il dataset deve avere lo stesso formato di quello dei politici.
#     """
#     print(f"  Loading youth text corpus: {text_json_path}")
#     chunks = load_politician_chunks(text_json_path)
#     print(f"  Chunks: {len(chunks)}")
#     hypotheses = list(CATEGORIES.values())
#     # Applica lo stesso hard-threshold usato per i politici
#     scores = _classify_chunks(classifier, chunks, hypotheses)
#     dist   = scores.mean(axis=0)
#     # In questo caso NON forziamo 'propaganda' e 'altro' a zero, perché vogliamo
#     # misurare quanto "rumore" fanno anche i giovani sui social.
#     return dist / dist.sum()
# ──────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
#  NLI INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

_classifier_cache = None

def _get_classifier():
    global _classifier_cache
    if _classifier_cache is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"  Loading NLI model: {NLI_MODEL}  (device={'cuda' if device==0 else 'cpu'})")
        _classifier_cache = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=device,
        )
    return _classifier_cache


def _classify_chunks(classifier, chunks: list[str], hypotheses: list[str], threshold: float = 0.5) -> np.ndarray:
    """
    Usa hard-thresholding: se il post non supera 0.5 in nessun tema,
    viene automaticamente catalogato in 'altro'.
    """
    scores = np.zeros((len(chunks), len(hypotheses)))
    
    altro_hyp = "Questo testo tratta di argomenti generici, lifestyle, o argomenti non politici."
    altro_idx = hypotheses.index(altro_hyp) if altro_hyp in hypotheses else None

    for i in range(0, len(chunks), NLI_BATCH_SIZE):
        batch = chunks[i:i + NLI_BATCH_SIZE]
        results = classifier(batch, candidate_labels=hypotheses, multi_label=True)
        if isinstance(results, dict):
            results = [results]
        for j, result in enumerate(results):
            label_to_score = dict(zip(result["labels"], result["scores"]))
            chunk_has_match = False
            for h_idx, hyp in enumerate(hypotheses):
                score = label_to_score.get(hyp, 0.0)
                if score >= threshold:
                    scores[i + j, h_idx] = score
                    if h_idx != altro_idx:
                        chunk_has_match = True
                else:
                    scores[i + j, h_idx] = 0.0
            
            if not chunk_has_match and altro_idx is not None:
                scores[i + j, altro_idx] = 1.0

    return scores

# ═══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def agenda_alignment_score(p: np.ndarray, q: np.ndarray) -> float:
    return float(1.0 - jensenshannon(p, q, base=2) ** 2)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agenda Alignment Score")
    parser.add_argument("--politician", "-p", type=str, help="Only run for this politician")
    parser.add_argument("--small-run", "-s", action="store_true", help="Quick smoke-test")
    args = parser.parse_args()

    cat_keys       = list(CATEGORIES.keys())
    hypotheses     = list(CATEGORIES.values())
    meaningful_idx = list(range(len(cat_keys)))
    meaningful_keys = cat_keys

    # ── Youth distribution ───────────────────────────────────────────────────
    print("\n[ 1/3 ] Building youth distribution...")
    if YOUTH_SOURCE == "istat":
        print(f"  Source: ISTAT ground truth ({GROUND_TRUTH_PATH})")
        youth_dist = build_istat_distribution(GROUND_TRUTH_PATH, small_run=args.small_run)

    # ── [OPZIONALE] BLOCCO MAIN PER TESTI NON STRUTTURATI ──
    # elif YOUTH_SOURCE == "text":
    #     if not YOUTH_TEXT_PATH:
    #         print("ERROR: YOUTH_TEXT_PATH not set. Decomment it in configurations.")
    #         sys.exit(1)
    #     classifier = _get_classifier()
    #     youth_dist = build_text_distribution(YOUTH_TEXT_PATH, classifier)
    # ───────────────────────────────────────────────────────
    
    else:
        print(f"ERROR: Unknown YOUTH_SOURCE '{YOUTH_SOURCE}'.")
        sys.exit(1)

    print("\n  Youth distribution:")
    for k, v in zip(cat_keys, youth_dist):
        bar = "█" * int(v * 40)
        print(f"    {k:<15} {v:.3f}  {bar}")

    # ── Politician distributions ─────────────────────────────────────────────
    print("\n[ 2/3 ] Classifying politician content...")
    classifier   = _get_classifier()
    politicians  = {}

    politician_list = sorted(os.listdir(CONTENT_DIR))
    if args.politician:
        if args.politician in politician_list:
            politician_list = [args.politician]
        else:
            print(f"ERROR: Politician '{args.politician}' non trovato.")
            sys.exit(1)

    for politician in politician_list:
        politician_dir = os.path.join(CONTENT_DIR, politician)
        json_path      = os.path.join(politician_dir, f"{politician}.json")
        if not os.path.isfile(json_path):
            continue

        print(f"\n  → {politician}")
        chunks = load_politician_chunks(json_path)
        if args.small_run:
            chunks = chunks[:SMALL_RUN_CHUNKS]
        print(f"    Chunks: {len(chunks)}")

        if not chunks:
            continue

        scores = _classify_chunks(classifier, chunks, hypotheses)
        dist   = scores.mean(axis=0)
        
        meaningful_sum = dist[meaningful_idx].sum()
        dist_normalized = dist.copy()
        if meaningful_sum > 0:
            dist_normalized[meaningful_idx] /= meaningful_sum
        politicians[politician] = dist_normalized

        for k, v in zip(cat_keys, dist_normalized):
            bar = "█" * int(v * 40)
            print(f"    {k:<15} {v:.3f}  {bar}")

    # ── Alignment scores ─────────────────────────────────────────────────────
    print("\n[ 3/3 ] Computing Agenda Alignment Scores...")
    results = []
    for politician, pol_dist in politicians.items():
        score = agenda_alignment_score(
            youth_dist[meaningful_idx],
            pol_dist[meaningful_idx],
        )
        results.append({
            "politician":        politician,
            "alignment_score":   round(score, 4),
            "distribution":      {k: round(float(v), 4) for k, v in zip(cat_keys, pol_dist)},
            "noise_score":       round(float(pol_dist[cat_keys.index("propaganda")]) if "propaganda" in cat_keys else 0.0, 4),
            "altro_score":       round(float(pol_dist[cat_keys.index("altro")]) if "altro" in cat_keys else 0.0, 4),
        })

    results.sort(key=lambda x: x["alignment_score"], reverse=True)

    print(f"\n{'═'*70}")
    print("  AGENDA ALIGNMENT SCORES")
    print(f"{'═'*70}")
    for r in results:
        bar   = "█" * int(r["alignment_score"] * 40)
        print(f"  {r['politician']:<25} {r['alignment_score']:.4f}  {bar}  [rumore: {(r.get('noise_score', 0) + r.get('altro_score', 0)):.2f}]")
    print(f"{'═'*70}")

    # ── Export ───────────────────────────────────────────────────────────────
    output = {
        "categories":           cat_keys,
        "meaningful_categories": meaningful_keys,
        "youth_distribution":   {k: round(float(v), 4) for k, v in zip(cat_keys, youth_dist)},
        "politicians":           results,
    }
    out_path = os.path.join(OUTPUT_DIR, "alignment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    rows = []
    for r in results:
        row = {"politician": r["politician"], "alignment_score": r["alignment_score"]}
        row.update({f"pol_{k}": v for k, v in r["distribution"].items()})
        row.update({f"youth_{k}": round(float(youth_dist[i]), 4) for i, k in enumerate(cat_keys)})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "alignment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}\n")

if __name__ == "__main__":
    main()