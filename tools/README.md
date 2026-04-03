# Instagram Scraper

Scarica i post piu' recenti di un profilo Instagram in un intervallo di giorni configurabile. Per ogni post salva i media e i metadata (likes, commenti, caption) in JSON. Opzionalmente esegue OCR sulle immagini e trascrizione audio sui video.

Sono inclusi due script:
- `scraper_plus.py` — download completo con OCR (pytesseract) e trascrizione audio (Whisper)
- `scraper.py` — solo download e metadata in JSON

---

## Setup

### 1. Creare e attivare la virtual environment

Linux/macOS:
```
python -m venv venv
source venv/bin/activate
```

Windows:
```
python -m venv venv
venv\Scripts\activate
```

### 2. Installare le dipendenze Python

```
pip install -r requirements.txt
```

### 3. Dipendenze di sistema

ffmpeg e tesseract non si installano via pip, vanno installati separatamente.

Linux:
```
sudo apt install ffmpeg tesseract-ocr tesseract-ocr-ita
```

macOS:
```
brew install ffmpeg tesseract tesseract-lang
```

### 4. File .env

Crea un file `.env` nella root del progetto. Questi sono i cookie della sessione Instagram autenticata, recuperabili da Chrome in: F12 > Application > Cookies > instagram.com

```
SESSION_ID=
CSRF_TOKEN=
DS_USER_ID=
MID=
```

### 5. Generare session.txt

```
python session.py
```

Verifica il login e salva la sessione in `session.txt`. Va rigenerato quando la sessione scade.

### 6. Configurare i profili

In cima a `scraper.py` o `scraper_plus.py`:

```python
USERNAME = "il_tuo_username"
TARGET_PROFILE = "profilo_da_scrapare"
DAYS = 7
```

---

## Utilizzo

```
python scraper.py
```
oppure
```
python scraper_plus.py
```
