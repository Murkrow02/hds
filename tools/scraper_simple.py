import instaloader
import os
import json
import time
import shutil
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configurazione di base ---
USERNAME = os.getenv("INSTA_USERNAME", "")
TARGET_PROFILE = "matteosalviniofficial"
START_DATE = "2026-04-09"  # Formato YYYY-MM-DD
END_DATE = ""           # Formato YYYY-MM-DD, lascia vuoto per oggi

# --- Parsing Argomenti ---
parser = argparse.ArgumentParser(description="Instagram Scraper Simple")
parser.add_argument("-t", "--target-profile", type=str, help="Profilo Instagram da scaricare")
args = parser.parse_args()

# Logic for Target Profile
if args.target_profile:
    TARGET_PROFILE = args.target_profile
    print(f"[CONFIG] Target profile impostato da ARGOMENTO: {TARGET_PROFILE}")
elif TARGET_PROFILE:
    TARGET_PROFILE = TARGET_PROFILE
    print(f"[CONFIG] Target profile letto da .env: {TARGET_PROFILE}")
else:
    print("[ERRORE] Nessun target profile fornito né tramite argomento né nel file .env")
    exit(1)

DATASET_DIR = os.path.join("dataset", TARGET_PROFILE)
os.makedirs(DATASET_DIR, exist_ok=True)

print(f"[CONFIG] Username Instagram: {USERNAME}")
print(f"[CONFIG] Intervallo Date: {START_DATE} -> {END_DATE if END_DATE else 'Oggi'}")
print(f"[CONFIG] Cartella Dataset: {DATASET_DIR}\n")

# --- Instaloader ---

L = instaloader.Instaloader(
    download_comments=False,
    save_metadata=False,
    dirname_pattern=os.path.join("dataset", "{target}", "_temp_{shortcode}"),
    filename_pattern="{shortcode}_{filename}",
    sleep=True,
    quiet=True
)

try:
    if os.path.exists("session.txt"):
        L.load_session_from_file(USERNAME, "session.txt")
    elif os.path.exists("tools/session.txt"):
        L.load_session_from_file(USERNAME, "tools/session.txt")
    else:
        raise FileNotFoundError
    print(f"Sessione caricata. Profilo: {TARGET_PROFILE}")
except FileNotFoundError:
    print("Session file non trovato, faccio login...")
    L.interactive_login(USERNAME)
    L.save_session_to_file("session.txt")

profile = instaloader.Profile.from_username(L.context, TARGET_PROFILE)
print(f"Post totali sul profilo: {profile.mediacount}\n")

# --- Utility ---

def get_folder_id(base_path, date_str):
    if not os.path.exists(base_path):
        return f"{date_str}_001"
    existing = [
        d for d in os.listdir(base_path)
        if d.startswith(date_str)
        and os.path.isdir(os.path.join(base_path, d))
        and not d.startswith("_temp")
    ]
    return f"{date_str}_{len(existing) + 1:03d}"

def rename_media_files(temp_folder, final_folder, folder_id):
    media_files = []
    counter = 1
    for filename in sorted(os.listdir(temp_folder)):
        full_path = os.path.join(temp_folder, filename)
        if filename.endswith(".txt") or filename.endswith(".json.xz") or filename.endswith(".json"):
            if os.path.exists(full_path):
                os.remove(full_path)
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".mp4", ".webp"]:
            new_name = f"{folder_id}_{counter}{ext}"
            new_path = os.path.join(final_folder, new_name)
            os.rename(full_path, new_path)
            media_files.append(new_name)
            counter += 1
    return media_files

# --- Main ---

start_date = datetime.strptime(START_DATE, "%Y-%m-%d").date()
end_date = datetime.strptime(END_DATE, "%Y-%m-%d").date() if END_DATE else datetime.now().date()

count = 0
old_streak = 0

print(f"Scarico media: {TARGET_PROFILE} | intervallo ({start_date} -> {end_date})")

for post in profile.get_posts():
    post_date = post.date_utc.date()

    if post_date > end_date:
        continue
    if post_date < start_date:
        old_streak += 1
        if old_streak >= 5:
            break
        continue

    old_streak = 0
    folder_id = get_folder_id(DATASET_DIR, str(post_date))
    final_folder = os.path.join(DATASET_DIR, folder_id)
    temp_folder = os.path.join(DATASET_DIR, f"_temp_{post.shortcode}")

    os.makedirs(final_folder, exist_ok=True)

    status = "OK"
    try:
        L.download_post(post, TARGET_PROFILE)
    except Exception as e:
        status = f"ERRORE: {e}"

    if os.path.exists(temp_folder):
        media_files = rename_media_files(temp_folder, final_folder, folder_id)
        try:
            shutil.rmtree(temp_folder)
        except Exception:
            pass
        
        # Salviamo info temporanee per il transcriber
        info = {
            "folder_id": folder_id,
            "caption": post.caption or "",
            "type": "video" if post.is_video else "image"
        }
        with open(os.path.join(final_folder, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        
        count += 1
        print(f"[{count:>2}] {post_date} {info['type']:<9} {status:<6}")

    time.sleep(3)

print(f"\nDownload completato. {count} post scaricati.")