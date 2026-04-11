# Text Transcriber

Questo tool si occupa di estrarre testo dai media scaricati dallo scraper di Instagram. Applica OCR sulle immagini e trascrizione audio (tramite Whisper) sui video.

## Setup

1. **Attivare il virtual environment** (stesso dello scraper o uno dedicato)
2. **Installare le dipendenze Python**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dipendenze di sistema**:
   Devono essere installati `ffmpeg` e `tesseract-ocr`.
   - **Linux**: `sudo apt install ffmpeg tesseract-ocr tesseract-ocr-ita`
   - **macOS**: `brew install ffmpeg tesseract tesseract-lang`

## Strumenti inclusi

### 1. `transcribe.py`
Processa una cartella di un profilo scaricato, esegue OCR/Whisper, e crea un file `.json` finale per ogni post.
Alla fine del processo, unisce automaticamente tutti i file JSON individuali raggruppandoli in `data/content/<profilo>/<profilo>.json`.

```bash
python transcribe.py <profilo>
```
Esempio:
```bash
python transcribe.py matteosalviniofficial
```
> **Nota**: Legge i dati da `data/content/<profilo>/` e genera o aggiorna i JSON nella stessa cartella.
