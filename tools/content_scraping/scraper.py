import instaloader
import os
import json
import time
import shutil
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

# --- Configurazione ---
USERNAME = "your_instagram_username"  # Sostituisci con il tuo username Instagram
TARGET_PROFILE = "target_profile_username"  # Sostituisci con il profilo da cui vuoi scaricare i post
DAYS = 7
DATASET_DIR = os.path.join("dataset", TARGET_PROFILE)

# --- Date ---
today = datetime.now().date()
cutoff_date = today - timedelta(days=DAYS)

os.makedirs(DATASET_DIR, exist_ok=True)

print(f"Scarico post: {TARGET_PROFILE} | ultimi {DAYS} giorni ({cutoff_date} -> {today})")

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
    L.load_session_from_file(USERNAME, "session.txt")
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
        if filename.endswith(".txt"):
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


def get_post_type(post, media_files):
    if post.is_video:
        return "video"
    if len(media_files) > 1:
        return "album"
    return "immagine"


def truncate(text, length=45):
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:length] + "..." if len(text) > length else text


# --- Main ---

count = 0
old_streak = 0

for post in profile.get_posts():
    post_date = post.date_utc.date()

    if post_date < cutoff_date:
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
        status = f"ERRORE download: {e}"

    media_files = []
    if os.path.exists(temp_folder):
        media_files = rename_media_files(temp_folder, final_folder, folder_id)
        try:
            os.rmdir(temp_folder)
        except Exception:
            pass
    else:
        status = "ERRORE cartella temp"

    metadata = {
        "profile": TARGET_PROFILE,
        "shortcode": post.shortcode,
        "folder_id": folder_id,
        "date": str(post_date),
        "likes": post.likes,
        "comments": post.comments,
        "is_video": post.is_video,
        "caption": post.caption or "",
        "media_files": media_files
    }
    with open(os.path.join(final_folder, f"{folder_id}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    count += 1
    post_type = get_post_type(post, media_files)
    print(f"[{count:>2}]  {post_date}  {post_type:<9}  {status:<6}  \"{truncate(post.caption)}\"")

    time.sleep(3)

print(f"\nCompletato. {count} post scaricati.")