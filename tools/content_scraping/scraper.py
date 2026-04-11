import instaloader
import os
import json
import time
import shutil
import argparse
from datetime import datetime, timedelta

from requests.cookies import RequestsCookieJar
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SESSION_FILE = os.path.join(SCRIPT_DIR, "session.txt")


# --- Session Management ---

def load_or_create_session(loader):
    """
    Tenta di caricare la sessione da session.txt.
    Se non esiste o è scaduta, la rigenera dai cookie nel .env.
    """
    username = _try_load_session(loader)
    if username:
        return username

    print("[SESSION] session.txt non trovato o scaduto, rigenero dai cookie .env...")
    username = _create_session_from_cookies(loader)
    return username


def _try_load_session(loader):
    """Prova a caricare session.txt e verifica che sia ancora valida."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        # Serve uno username qualsiasi per caricare il file, verrà sovrascritto dal test
        loader.load_session_from_file("_", SESSION_FILE)
        username = loader.test_login()
        if username:
            print(f"[SESSION] Sessione caricata. Login: {username}")
            return username
        print("[SESSION] Sessione scaduta.")
        return None
    except Exception as e:
        print(f"[SESSION] Errore caricamento sessione: {e}")
        return None


def _create_session_from_cookies(loader):
    """Crea una nuova sessione da cookie Instagram nel .env."""
    session_id = os.getenv("SESSION_ID")
    csrf_token = os.getenv("CSRF_TOKEN")
    ds_user_id = os.getenv("DS_USER_ID")
    mid = os.getenv("MID")

    if not all([session_id, csrf_token, ds_user_id, mid]):
        print("[ERRORE] Variabili mancanti nel .env. Servono: SESSION_ID, CSRF_TOKEN, DS_USER_ID, MID")
        print("         Recuperali da Chrome: F12 > Application > Cookies > instagram.com")
        exit(1)

    cookies = {
        "sessionid": session_id,
        "csrftoken": csrf_token,
        "ds_user_id": ds_user_id,
        "mid": mid,
    }

    jar = RequestsCookieJar()
    for name, value in cookies.items():
        jar.set(name, value, domain=".instagram.com", path="/")

    loader.context._session.cookies.update(jar)

    username = loader.test_login()
    if not username:
        print("[ERRORE] Login fallito. Controlla che i cookie nel .env siano validi e non scaduti.")
        exit(1)

    print(f"[SESSION] Login riuscito: {username}")
    loader.context.username = username
    loader.save_session_to_file(SESSION_FILE)
    print(f"[SESSION] Sessione salvata in {SESSION_FILE}")
    return username


# --- Utility ---

def get_folder_id(base_path, date_str):
    """Genera un ID cartella univoco per la data (es. 2026-04-10_003)."""
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
    """Rinomina i media dalla cartella temp alla cartella finale con naming coerente."""
    media_files = []
    counter = 1
    for filename in sorted(os.listdir(temp_folder)):
        full_path = os.path.join(temp_folder, filename)
        # Rimuovi file non-media (metadata instaloader, ecc.)
        if filename.endswith((".txt", ".json.xz", ".json")):
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Instagram Scraper — scarica media e metadati da un profilo Instagram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Esempi:
  python scraper.py matteosalviniofficial
  python scraper.py matteosalviniofficial --start 2026-04-01
  python scraper.py matteosalviniofficial --start 2026-04-01 --end 2026-04-11
"""
    )
    parser.add_argument(
        "profile",
        type=str,
        help="Username del profilo Instagram da scaricare (obbligatorio)"
    )
    parser.add_argument(
        "-s", "--start",
        type=str,
        default=None,
        help="Data inizio in formato YYYY-MM-DD (default: 7 giorni fa)"
    )
    parser.add_argument(
        "-e", "--end",
        type=str,
        default=None,
        help="Data fine in formato YYYY-MM-DD (default: oggi)"
    )
    return parser.parse_args()


def parse_date(date_str, label):
    """Parsa una stringa data in formato YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERRORE] Formato data non valido per --{label}: '{date_str}'. Usa YYYY-MM-DD.")
        exit(1)


def main():
    args = parse_args()

    target_profile = args.profile
    end_date = parse_date(args.end, "end") if args.end else datetime.now().date()
    start_date = parse_date(args.start, "start") if args.start else (end_date - timedelta(days=7))

    if start_date > end_date:
        print(f"[ERRORE] La data inizio ({start_date}) è successiva alla data fine ({end_date}).")
        exit(1)

    dataset_dir = os.path.join(PROJECT_ROOT, "data", "content", target_profile)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"[CONFIG] Profilo:    {target_profile}")
    print(f"[CONFIG] Intervallo: {start_date} -> {end_date}")
    print(f"[CONFIG] Output:     {dataset_dir}\n")

    # --- Instaloader setup ---
    L = instaloader.Instaloader(
        download_comments=False,
        save_metadata=False,
        dirname_pattern=os.path.join(dataset_dir, "_temp_{shortcode}"),
        filename_pattern="{shortcode}_{filename}",
        sleep=True,
        quiet=True,
    )

    load_or_create_session(L)

    profile = instaloader.Profile.from_username(L.context, target_profile)
    print(f"[INFO] Post totali sul profilo: {profile.mediacount}\n")

    # --- Download loop ---
    count = 0
    old_streak = 0

    print(f"Scarico media: {target_profile} | intervallo ({start_date} -> {end_date})")

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
        folder_id = get_folder_id(dataset_dir, str(post_date))
        final_folder = os.path.join(dataset_dir, folder_id)
        temp_folder = os.path.join(dataset_dir, f"_temp_{post.shortcode}")

        os.makedirs(final_folder, exist_ok=True)

        status = "OK"
        try:
            L.download_post(post, target_profile)
        except Exception as e:
            status = f"ERRORE: {e}"

        if os.path.exists(temp_folder):
            media_files = rename_media_files(temp_folder, final_folder, folder_id)
            try:
                shutil.rmtree(temp_folder)
            except Exception:
                pass

            # Salva info.json per il transcriber
            info = {
                "folder_id": folder_id,
                "caption": post.caption or "",
                "type": "video" if post.is_video else "image",
            }
            with open(os.path.join(final_folder, "info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)

            count += 1
            print(f"[{count:>2}] {post_date}  {info['type']:<9} {status:<6}")

        time.sleep(3)

    print(f"\nCompletato. {count} post scaricati.")


if __name__ == "__main__":
    main()