import os
import ffmpeg
from faster_whisper import WhisperModel

# --- CONFIGURATION FOR APPLE M1 ---
DATA_DIRECTORY = "data"
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEVICE = "cpu"          
COMPUTE_TYPE = "int8"   

# Initialize Model
print(f"--- Initializing Whisper {MODEL_SIZE} on Apple Silicon ---")
model = WhisperModel(
    MODEL_SIZE, 
    device="cpu", 
    compute_type="int8", 
    cpu_threads=8,       
    num_workers=2       
)

def process_files():
    file_count = 0
    
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        video_files = [f for f in files if f.lower().endswith(".mp4")]
        
        if video_files:
            print(f"\n📂 Folder: {root}")
            
        for file in video_files:
            file_count += 1
            video_path = os.path.join(root, file)
            base_name = os.path.splitext(video_path)[0]
            
            audio_output = f"{base_name}-audio.wav"
            text_output = f"{base_name}-transcribed.txt"

            print(f"  📄 [{file_count}] {file}")

            # 1. Extract Audio
            if not os.path.exists(audio_output):
                print(f"    -> 🎵 Extracting audio...")
                try:
                    (
                        ffmpeg
                        .input(video_path)
                        .output(audio_output, ar='16000', ac=1, format='wav')
                        .overwrite_output()
                        .run(quiet=True)
                    )
                except ffmpeg.Error:
                    print(f"    ❌ ERROR: FFmpeg failed on {file}")
                    continue

            # 2. Transcribe
            if not os.path.exists(text_output):
                print(f"    -> ✍️  Transcribing (No Timestamps)...")
                segments, info = model.transcribe(audio_output, beam_size=5)

                with open(text_output, "w", encoding="utf-8") as f:
                    for segment in segments:
                        # Clean the text to remove leading/trailing whitespace
                        clean_text = segment.text.strip()
                        
                        # Live print for progress
                        print(f"       {clean_text}")
                        
                        # Write just the text followed by a space or newline
                        # Using a space creates a continuous paragraph; \n creates a list.
                        f.write(clean_text + " ") 
                
                print(f"\n    -> ✅ Saved: {os.path.basename(text_output)}")
            else:
                print(f"    -> ✅ Skipping: Transcription already exists.")

if __name__ == "__main__":
    if os.path.exists(DATA_DIRECTORY):
        process_files()
        print("\n🎉 Done! All files processed.")
    else:
        print(f"Directory '{DATA_DIRECTORY}' not found.")
