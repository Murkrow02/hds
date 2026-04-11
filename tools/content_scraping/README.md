# Instagram Scraper

Scarica i post di un profilo Instagram in un intervallo di date configurabile. Per ogni post salva i file media (immagini, video) e un file `info.json` con caption e tipo di contenuto.

I dati scaricati vengono salvati in `data/content/<profilo>/` con una cartella per ogni post, nel formato `<data>_<numero>`.

---

## Setup

### 1. Creare e attivare la virtual environment

Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Installare le dipendenze Python

```bash
pip install -r requirements.txt
```

### 3. File `.env`

Crea un file `.env` nella root del progetto con i cookie della sessione Instagram autenticata.

Recuperali da Chrome: `F12 > Application > Cookies > instagram.com`

```env
SESSION_ID=...
CSRF_TOKEN=...
DS_USER_ID=...
MID=...
```

> La sessione viene generata automaticamente al primo avvio dello scraper e salvata in `session.txt`. Quando scade, viene rigenerata automaticamente dai cookie nel `.env`.

---

## Utilizzo

```bash
python scraper.py <profilo> [--start YYYY-MM-DD] [--end YYYY-MM-DD]
```

### Argomenti

| Argomento | Tipo | Obbligatorio | Default | Descrizione |
|---|---|---|---|---|
| `profilo` | posizionale | ✅ | — | Username del profilo Instagram |
| `--start`, `-s` | flag | ❌ | 7 giorni fa | Data di inizio (formato `YYYY-MM-DD`) |
| `--end`, `-e` | flag | ❌ | oggi | Data di fine (formato `YYYY-MM-DD`) |

### Esempi

```bash
# Ultimi 7 giorni (default)
python scraper.py matteosalviniofficial

# Da una data specifica a oggi
python scraper.py matteosalviniofficial --start 2026-04-01

# Intervallo specifico
python scraper.py matteosalviniofficial --start 2026-04-01 --end 2026-04-11
```

---

## Struttura output

```
data/content/<profilo>/
├── 2026-04-09_001/
│   ├── 2026-04-09_001_1.jpg
│   ├── 2026-04-09_001_2.jpg
│   └── info.json
├── 2026-04-10_001/
│   ├── 2026-04-10_001_1.mp4
│   └── info.json
└── 2026-04-10_002/
    ├── 2026-04-10_002_1.jpg
    └── info.json
```

Ogni `info.json` contiene:
```json
{
    "folder_id": "2026-04-10_001",
    "caption": "Testo della caption del post...",
    "type": "video"
}
```
