import os
import json
import shutil
import argparse
import ffmpeg
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
OCR_LANG = "ita+eng"

# Faster-Whisper Configuration (Apple Silicon Optimized)
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
ENABLE_OCR = True
ENABLE_TRANSCRIPTION = True

# --- Parsing Argomenti ---
parser = argparse.ArgumentParser(description="Instagram Media Transcriber & OCR")
parser.add_argument("profile", type=str, help="Profilo Instagram da elaborare (obbligatorio)")
args = parser.parse_args()

TARGET_PROFILE = args.profile
print(f"[CONFIG] Target profile impostato da argomenti: {TARGET_PROFILE}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "content", TARGET_PROFILE)

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# --- Utility ---

def clean_text(text):
    if not text: return ""
    return " ".join(text.split()).strip()

def run_ocr_on_image(image_path, lang=OCR_LANG):
    try:
        img = Image.open(image_path).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(img, lang=lang)
        return clean_text(text)
    except Exception:
        return ""

def transcribe_video(video_path, folder_path, model):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(folder_path, f"{base_name}_audio.wav")
    try:
        # 1. Extract Audio using ffmpeg-python pipeline
        if not os.path.exists(audio_path):
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, ar='16000', ac=1, format='wav')
                .overwrite_output()
                .run(quiet=True)
            )
        
        # 2. Transcribe using faster-whisper pipeline
        segments, info = model.transcribe(audio_path, beam_size=5, language="it")
        
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        return " ".join(text_parts).strip()
    except Exception as e:
        print(f"    [ERRORE] Trascrizione video {os.path.basename(video_path)}: {e}")
        return ""
    finally:
        if os.path.exists(audio_path): 
            os.remove(audio_path)

def combine_json_files(profile_name, dataset_dir):
    all_posts_data = []
    subdirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    print(f"\n[INFO] Creazione file JSON aggregato per {profile_name}...")

    for subdir in subdirs:
        if subdir.startswith("_temp"):
            continue
            
        json_file_path = os.path.join(dataset_dir, subdir, f"{subdir}.json")
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_posts_data.append(data)
            except Exception as e:
                print(f"[ERRORE] Lettura di {json_file_path}: {e}")
        else:
            # Fallback per consistenza
            for file in os.listdir(os.path.join(dataset_dir, subdir)):
                if file.endswith(".json") and file != "info.json":
                    try:
                        with open(os.path.join(dataset_dir, subdir, file), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_posts_data.append(data)
                            break
                    except Exception:
                        continue

    if all_posts_data:
        output_file = os.path.join(dataset_dir, f"{profile_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_posts_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Creato file unico: {output_file}")
        print(f"[INFO] Totale post inclusi: {len(all_posts_data)}")
    else:
        print("[AVVISO] Nessun JSON trovato da combinare.")


# --- Main ---

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERRORE] La cartella {DATASET_DIR} non esiste.")
        return

    print(f"[INFO] Inizializzazione Faster-Whisper {MODEL_SIZE}...")
    whisper_model = WhisperModel(
        MODEL_SIZE, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE, 
        cpu_threads=8,       
        num_workers=2       
    ) if ENABLE_TRANSCRIPTION else None

    folders = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith("_temp")])
    
    print(f"[INFO] Inizio elaborazione: {len(folders)} cartelle trovate per {TARGET_PROFILE}\n")
    
    count = 0
    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        info_file = os.path.join(folder_path, "info.json")
        final_json = os.path.join(folder_path, f"{folder}.json")

        if os.path.exists(final_json):
            continue

        if not os.path.exists(info_file):
            continue

        with open(info_file, "r", encoding="utf-8") as f:
            post_info = json.load(f)

        print(f"[{count+1:>2}] Elaborazione: {folder}")
        
        media_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.webp'))]
        extracted_text_list = []

        for media_file in media_files:
            media_path = os.path.join(folder_path, media_file)
            ext = os.path.splitext(media_file)[1].lower()
            
            if ENABLE_OCR and ext in [".jpg", ".jpeg", ".png", ".webp"]:
                print(f"    -> Trascrizione immagine...")
                text = run_ocr_on_image(media_path)
                if text: extracted_text_list.append(text)
                
            if ENABLE_TRANSCRIPTION and ext == ".mp4":
                print(f"    -> Trascrizione video...")
                text = transcribe_video(media_path, folder_path, whisper_model)
                if text: extracted_text_list.append(text)

        # Generazione JSON finale
        post_data = {
            "folder_id": post_info["folder_id"],
            "caption": post_info["caption"],
            "type": post_info["type"],
            "text": " ".join(extracted_text_list).strip(),
            "language": "it"
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=4, ensure_ascii=False)
        
        count += 1

    print(f"\n[INFO] Completato. {count} nuove cartelle elaborate.")
    
    combine_json_files(TARGET_PROFILE, DATASET_DIR)

if __name__ == "__main__":
    main()
