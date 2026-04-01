import instaloader
import os
import json
import time
import shutil
import subprocess
from datetime import datetime, timedelta

import pytesseract
from PIL import Image, ImageOps, ImageFilter
import whisper
from dotenv import load_dotenv

load_dotenv()

# --- Configurazione ---
USERNAME = ""  # Inserisci il tuo username Instagram qui (necessario per sessione)
TARGET_PROFILE = ""  # Inserisci il profilo Instagram da scaricare
DAYS = 7
DATASET_DIR = os.path.join("dataset", TARGET_PROFILE)

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

OCR_LANG = "ita+eng"
WHISPER_MODEL = "small"
ENABLE_OCR = True
ENABLE_TRANSCRIPTION = True

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


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split()).strip()


def preprocess_image_for_ocr(image_path):
    img = Image.open(image_path).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def run_ocr_on_image(image_path, lang=OCR_LANG):
    try:
        img = preprocess_image_for_ocr(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = []
        for c in data.get("conf", []):
            try:
                c = float(c)
                if c >= 0:
                    confidences.append(c)
            except Exception:
                pass
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else None
        return {
            "type": "image",
            "source_file": os.path.basename(image_path),
            "text": clean_text(text),
            "avg_confidence": avg_conf
        }
    except Exception as e:
        return {
            "type": "image",
            "source_file": os.path.basename(image_path),
            "error": str(e),
            "text": ""
        }


def extract_audio_from_video(video_path, audio_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg non trovato nel PATH di sistema.")
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path, language="it", fp16=False)
    return {
        "text": clean_text(result.get("text", "")),
        "segments": [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": clean_text(seg.get("text", ""))
            }
            for seg in result.get("segments", [])
        ]
    }


def transcribe_video(video_path, folder_path, model):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(folder_path, f"{base_name}_audio.wav")
    try:
        extract_audio_from_video(video_path, audio_path)
        transcript = transcribe_audio(audio_path, model)
        out = {
            "type": "video_audio",
            "source_file": os.path.basename(video_path),
            "transcript_text": transcript["text"],
            "segments": transcript["segments"]
        }
    except Exception as e:
        out = {
            "type": "video_audio",
            "source_file": os.path.basename(video_path),
            "error": str(e),
            "transcript_text": "",
            "segments": []
        }
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return out


def process_media(final_folder, folder_id, media_files, whisper_model=None):
    ocr_results = []
    transcript_results = []
    for media_file in media_files:
        media_path = os.path.join(final_folder, media_file)
        ext = os.path.splitext(media_file)[1].lower()
        if ENABLE_OCR and ext in [".jpg", ".jpeg", ".png", ".webp"]:
            ocr_results.append(run_ocr_on_image(media_path))
        if ENABLE_TRANSCRIPTION and ext == ".mp4":
            transcript_results.append(transcribe_video(media_path, final_folder, whisper_model))
    with open(os.path.join(final_folder, f"{folder_id}_ocr.json"), "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=4, ensure_ascii=False)
    with open(os.path.join(final_folder, f"{folder_id}_transcript.json"), "w", encoding="utf-8") as f:
        json.dump(transcript_results, f, indent=4, ensure_ascii=False)


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

whisper_model = whisper.load_model(WHISPER_MODEL) if ENABLE_TRANSCRIPTION else None

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

    try:
        process_media(final_folder, folder_id, media_files, whisper_model)
    except Exception as e:
        status = f"ERRORE processing: {e}"

    post_type = get_post_type(post, media_files)
    caption = truncate(post.caption)
    count += 1

    print(f"[{count:>2}]  {post_date}  {post_type:<9}  {status:<6}  \"{caption}\"")

    metadata = {
        "profile": TARGET_PROFILE,
        "shortcode": post.shortcode,
        "folder_id": folder_id,
        "date": str(post_date),
        "likes": post.likes,
        "comments": post.comments,
        "is_video": post.is_video,
        "caption": post.caption or "",
        "media_files": media_files,
        "ocr_file": f"{folder_id}_ocr.json",
        "transcript_file": f"{folder_id}_transcript.json"
    }
    with open(os.path.join(final_folder, f"{folder_id}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    time.sleep(3)

print(f"\nCompletato. {count} post scaricati.")