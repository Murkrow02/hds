"""
Agenda Alignment Score
======================
Measures how well politician social content aligns with youth concerns.

Pipeline:
  1. Define concern categories (edit CATEGORIES below — no retraining needed)
  2. Build youth distribution:
       - ISTAT mode:  parse ground_truth_multiyr.json (structured survey data)
       - Text mode:   run NLI on any unstructured text corpus (same as politicians)
  3. Run NLI zero-shot on politician chunks → politician distribution
  4. Compute JSD → Agenda Alignment Score per politician

To add/remove categories: edit CATEGORIES. Rerun. Done.
To use a different youth source: set YOUTH_SOURCE = "text" and point YOUTH_TEXT_PATH
to a JSON file with the same format as politician data.
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
#  Add new lines here to extend (e.g. social life, housing, mental health...).
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    # Broadened vocabulary: captures "transizione ecologica", "green deal",
    # "rinnovabili", "siccità", not just the literal word "ambiente"
    "clima":        "Questo testo esprime preoccupazione per il cambiamento climatico, "
                    "la transizione ecologica, le energie rinnovabili o l'emergenza ambientale.",

    # Anchored to lived experience (precariato, affitti, stipendi bassi)
    # not generic economic commentary
    "economia":     "Questo testo descrive difficoltà economiche vissute in prima persona, "
                    "come precarietà lavorativa, disoccupazione giovanile, salari bassi o "
                    "costo della vita insostenibile.",

    # Scoped to progressive civil rights to avoid "diritto alla sicurezza" false positives
    "diritti":      "Questo testo tratta di diritti civili, pari opportunità, parità di genere, "
                    "diritti LGBTQ+ o lotta alla discriminazione.",

    # Unchanged — well-scoped already
    "sicurezza":    "Questo testo tratta di criminalità, sicurezza urbana, ordine pubblico "
                    "o paura della violenza.",

    # Key fix: "esprime" forces the model to look for experiential disillusionment,
    # not political attacks framed as institutional critique
    "sfiducia":     "Questo testo esprime disincanto personale, astensionismo o sfiducia "
                    "verso i partiti politici e le istituzioni democratiche.",

    "immigrazione": "Questo testo tratta di immigrazione, politiche migratorie, "
                    "accoglienza o integrazione dei migranti.",

    # Dump category: absorbs partisan attacks, propaganda, electoral slogans.
    # Chunks scoring high here are politically noisy and less meaningful for alignment.
    # These scores are EXCLUDED from the youth-side distribution (youth don't post propaganda).
    # On the politician side they act as a noise sink during normalization.
    "_propaganda":  "Questo testo contiene attacchi politici contro avversari, "
                    "propaganda elettorale o polemiche di partito senza proposta concreta.",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Set these for a quick smoke-test: uses only SMALL_RUN_CHUNKS chunks per politician
# and only SMALL_RUN_ARCHETYPES archetypes from the ground truth.
SMALL_RUN_CHUNKS      = 30    # chunks per politician
SMALL_RUN_ARCHETYPES  = 10    # archetypes from ground truth

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTENT_DIR    = os.path.join(BASE_DIR, "data", "content")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# Youth source: "istat" (structured ground truth) or "text" (unstructured corpus)
YOUTH_SOURCE       = "istat"
GROUND_TRUTH_PATH  = os.path.join(PROCESSED_DIR, "ground_truth_multiyr.json")
YOUTH_TEXT_PATH    = None   # set to a politician-format JSON if YOUTH_SOURCE = "text"

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Chunking (mirrors classificator.py small-dataset settings)
CHUNK_MAX_WORDS = 150
CHUNK_MIN_WORDS = 30
CHUNK_OVERLAP   = 30

# Batch size for NLI inference — lower if OOM
NLI_BATCH_SIZE = 8

# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING  (shared with classificator.py logic)
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

# Keywords that appear in profile_description for each ISTAT dimension
_ISTAT_FLAGS = {
    "clima":     "fortemente preoccupato per il cambiamento climatico",
    "economia":  "insoddisfazione economica",
    "diritti":   "diritti civili",
    "sicurezza": "teme la criminalità",
    "sfiducia":  "sfiducia verso i partiti",
}


def build_istat_distribution(gt_path: str, small_run: bool = False) -> np.ndarray:
    """
    Derives a probability distribution over CATEGORIES from ground_truth_multiyr.json.

    For ISTAT dimensions (clima, economia, diritti, sicurezza, sfiducia): weight-sum
    of archetypes where the flag is True in profile_description.

    For categories without an ISTAT flag (e.g. immigrazione, or any custom ones you
    add): falls back to running NLI on the generated_statements of each archetype,
    weighted by weight_pct. This keeps new categories zero-effort to add.

    Returns a normalized numpy array aligned to list(CATEGORIES.keys()).
    """
    with open(gt_path, encoding="utf-8") as f:
        archetypes = json.load(f)

    if small_run:
        archetypes = archetypes[:SMALL_RUN_ARCHETYPES]
        print(f"  [SMALL_RUN] Using {SMALL_RUN_ARCHETYPES} archetypes out of full set")

    # Categories prefixed with "_" are noise sinks (e.g. _propaganda).
    # They are excluded from both distributions before JSD is computed.
    meaningful_keys = [k for k in CATEGORIES if not k.startswith("_")]
    cat_keys        = list(CATEGORIES.keys())
    dist            = np.zeros(len(cat_keys))

    # Split categories: those with a known ISTAT flag vs those needing NLI
    istat_mapped = {k: v for k, v in _ISTAT_FLAGS.items() if k in meaningful_keys}
    nli_needed   = [k for k in meaningful_keys if k not in istat_mapped]
    # _propaganda and other noise sinks always score 0 on the youth side (youth don't post propaganda)

    # ISTAT-mapped categories: direct flag lookup
    for archetype in archetypes:
        w    = archetype["weight_pct"] / 100.0
        desc = archetype["profile_description"].lower()
        for k, flag in istat_mapped.items():
            if flag.lower() in desc:
                dist[cat_keys.index(k)] += w

    # NLI-needed categories: classify generated_statements
    if nli_needed:
        print(f"  Categories without ISTAT flag — running NLI on statements: {nli_needed}")
        nli_hypotheses = [CATEGORIES[k] for k in nli_needed]
        classifier = _get_classifier()

        for archetype in archetypes:
            w          = archetype["weight_pct"] / 100.0
            statements = archetype.get("generated_statements", [])
            if not statements:
                continue
            scores = _classify_chunks(classifier, statements, nli_hypotheses)
            avg    = scores.mean(axis=0)  # (len(nli_needed),)
            for i, k in enumerate(nli_needed):
                dist[cat_keys.index(k)] += avg[i] * w

    # Normalize over meaningful categories only (noise sinks stay 0)
    meaningful_idx = [cat_keys.index(k) for k in meaningful_keys]
    total = dist[meaningful_idx].sum()
    if total == 0:
        raise ValueError("ISTAT distribution is all zeros — check _ISTAT_FLAGS keywords.")
    dist[meaningful_idx] /= total
    return dist


def build_text_distribution(text_json_path: str, classifier) -> np.ndarray:
    """Build youth distribution from an unstructured text JSON (same format as politician data)."""
    print(f"  Loading youth text corpus: {text_json_path}")
    chunks = load_politician_chunks(text_json_path)
    print(f"  Chunks: {len(chunks)}")
    hypotheses = list(CATEGORIES.values())
    scores = _classify_chunks(classifier, chunks, hypotheses)
    dist   = scores.mean(axis=0)
    return dist / dist.sum()

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


def _classify_chunks(classifier, chunks: list[str], hypotheses: list[str]) -> np.ndarray:
    """
    Returns shape (len(chunks), len(hypotheses)) with entailment probabilities.
    Uses multi_label=True so categories are independent (a chunk can address several).
    """
    scores = np.zeros((len(chunks), len(hypotheses)))
    for i in range(0, len(chunks), NLI_BATCH_SIZE):
        batch = chunks[i:i + NLI_BATCH_SIZE]
        results = classifier(batch, candidate_labels=hypotheses, multi_label=True)
        if isinstance(results, dict):
            results = [results]
        for j, result in enumerate(results):
            label_to_score = dict(zip(result["labels"], result["scores"]))
            for h_idx, hyp in enumerate(hypotheses):
                scores[i + j, h_idx] = label_to_score.get(hyp, 0.0)
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def agenda_alignment_score(p: np.ndarray, q: np.ndarray) -> float:
    """
    1 - JSD(P || Q).  Range [0, 1].  1 = perfect alignment, 0 = total divergence.
    JSD is squared here (proper divergence metric in [0,1]).
    """
    return float(1.0 - jensenshannon(p, q) ** 2)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agenda Alignment Score: Measures how well politician social content aligns with youth concerns.")
    parser.add_argument("--politician", "-p", type=str, help="Only run for this politician (folder name in data/content)")
    parser.add_argument("--small-run", "-s", action="store_true", help=f"Quick smoke-test: uses only {SMALL_RUN_CHUNKS} chunks per politician and {SMALL_RUN_ARCHETYPES} archetypes from ground truth.")
    args = parser.parse_args()

    cat_keys       = list(CATEGORIES.keys())
    hypotheses     = list(CATEGORIES.values())
    # Indices of meaningful categories (exclude noise sinks like _propaganda)
    meaningful_idx = [i for i, k in enumerate(cat_keys) if not k.startswith("_")]
    meaningful_keys = [cat_keys[i] for i in meaningful_idx]

    # ── Youth distribution ───────────────────────────────────────────────────
    print("\n[ 1/3 ] Building youth distribution...")
    if YOUTH_SOURCE == "istat":
        print(f"  Source: ISTAT ground truth ({GROUND_TRUTH_PATH})")
        youth_dist = build_istat_distribution(GROUND_TRUTH_PATH, small_run=args.small_run)
    elif YOUTH_SOURCE == "text":
        if not YOUTH_TEXT_PATH:
            print("ERROR: YOUTH_TEXT_PATH not set. Set it to a JSON file.")
            sys.exit(1)
        classifier = _get_classifier()
        youth_dist = build_text_distribution(YOUTH_TEXT_PATH, classifier)
    else:
        print(f"ERROR: Unknown YOUTH_SOURCE '{YOUTH_SOURCE}'. Use 'istat' or 'text'.")
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
            print(f"ERROR: Politician '{args.politician}' not found in {CONTENT_DIR}")
            sys.exit(1)

    for politician in politician_list:
        politician_dir = os.path.join(CONTENT_DIR, politician)
        json_path      = os.path.join(politician_dir, f"{politician}.json")
        if not os.path.isfile(json_path):
            print(f"  Skipping {politician} (no JSON found)")
            continue

        print(f"\n  → {politician}")
        chunks = load_politician_chunks(json_path)
        if args.small_run:
            chunks = chunks[:SMALL_RUN_CHUNKS]
            print(f"    Chunks: {len(chunks)} (SMALL_RUN, capped at {SMALL_RUN_CHUNKS})")
        else:
            print(f"    Chunks: {len(chunks)}")

        if not chunks:
            print("    No usable content, skipping.")
            continue

        scores = _classify_chunks(classifier, chunks, hypotheses)
        dist   = scores.mean(axis=0)
        # Normalize meaningful categories only; noise sinks kept as-is for diagnostics
        meaningful_sum = dist[meaningful_idx].sum()
        dist_normalized = dist.copy()
        if meaningful_sum > 0:
            dist_normalized[meaningful_idx] /= meaningful_sum
        politicians[politician] = dist_normalized

        for k, v in zip(cat_keys, dist_normalized):
            sink_marker = " [noise sink]" if k.startswith("_") else ""
            bar = "█" * int(v * 40)
            print(f"    {k:<15} {v:.3f}  {bar}{sink_marker}")

    # ── Alignment scores ─────────────────────────────────────────────────────
    print("\n[ 3/3 ] Computing Agenda Alignment Scores...")
    results = []
    for politician, pol_dist in politicians.items():
        # JSD only on meaningful categories (noise sinks excluded)
        score = agenda_alignment_score(
            youth_dist[meaningful_idx],
            pol_dist[meaningful_idx],
        )
        results.append({
            "politician":        politician,
            "alignment_score":   round(score, 4),
            # Full distribution saved (includes noise sinks for diagnostics)
            "distribution":      {k: round(float(v), 4) for k, v in zip(cat_keys, pol_dist)},
            # Propaganda/noise score — useful sanity check
            "noise_score":       round(float(np.mean([pol_dist[i] for i, k in enumerate(cat_keys)
                                                       if k.startswith("_")])), 4),
        })

    results.sort(key=lambda x: x["alignment_score"], reverse=True)

    print(f"\n{'═'*55}")
    print("  AGENDA ALIGNMENT SCORES")
    print(f"{'═'*55}")
    for r in results:
        bar   = "█" * int(r["alignment_score"] * 40)
        noise = r.get("noise_score", 0)
        print(f"  {r['politician']:<30} {r['alignment_score']:.4f}  {bar}  [noise: {noise:.3f}]")
    print(f"{'═'*55}")
    print("  High noise score = lots of propaganda/attacks → alignment less reliable for that politician")

    # ── Export ───────────────────────────────────────────────────────────────
    output = {
        "categories":           cat_keys,
        "meaningful_categories": meaningful_keys,  # excludes noise sinks
        "youth_distribution":   {k: round(float(v), 4) for k, v in zip(cat_keys, youth_dist)},
        "politicians":           results,
    }
    out_path = os.path.join(OUTPUT_DIR, "alignment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    # CSV with per-category breakdown
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
