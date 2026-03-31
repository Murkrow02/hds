# Human Data Science: The Representation Space

Analisi del **Semantic Gap** tra i bisogni delle nuove generazioni e la rappresentanza politica sui social media.

## Overview
Questo progetto per il corso di **Human Data Science** mira a quantificare la divergenza tra i dati statistici strutturati riguardanti le preoccupazioni dei giovani e i dati non strutturati prodotti dai leader politici. L'approccio si basa sulla creazione di un **Joint Embedding Space** dove proiettare entrambi i dataset per misurarne l'allineamento tematico.

### Metodologia Scientifica
*   **Modello di Misura:** Calcolo dell'**Agenda Alignment Score** tramite la *Jensen-Shannon Divergence (JSD)* tra le distribuzioni di probabilità dei temi.
*   **Semantic Mapping:** Proiezione dei contenuti in uno spazio vettoriale condiviso per superare il "Semantic Gap".
*   **Analisi Qualitativa:** Sentiment Analysis e Complexity Analysis (Indice di Gulpease) del parlato politico.

---

## Tools usati
| Task | Strumento / Modello |
| :--- | :--- |
| **Inference Engine** | Ollama / vLLM |
| **Trascrizione (STT)** | Faster-Whisper (Large-v3-turbo) |
| **Computer Vision** | Florence-2-base / PaddleOCR |
| **Embeddings** | BAAI/bge-m3 |
| **Vector DB** | ChromaDB / Qdrant |
| **Analisi Dati** | Python (Pandas, Scipy) |

---

## Multimodal Processing

### `tools/transcribe-data.py`
Questo script automatizza l'estrazione e la trascrizione dei contenuti multimediali.

**Funzionalità:**
1. **Recursive Scan:** Scansiona la directory `data/` alla ricerca di file `.mp4`.
2. **Audio Extraction:** Estrae la traccia audio in formato WAV utilizzando `FFmpeg`.
3. **Optimized Transcription:** Utilizza `faster-whisper-large-v3-turbo-ct2` per estrarre la trascrizione audio dai video.
4. **Output:** Genera file `-audio.wav` e `-transcribed.txt`.

#### Requisiti
* **FFmpeg**
* **Python Dependencies:** `pip install -r requirements.txt`

#### Utilizzo
```bash
# Assicurati di avere i video nella cartella data/
python tools/transcribe-data.py
```

---

