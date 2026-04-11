import os
import json
import sys

def combine_json_files(profile_name):
    dataset_dir = os.path.join("dataset", profile_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Errore: La cartella {dataset_dir} non esiste.")
        return

    all_posts_data = []
    
    # Iterate through all subdirectories (posts)
    subdirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    print(f"Analizzo {len(subdirs)} cartelle in {dataset_dir}...")

    for subdir in subdirs:
        # Ignore temp folders
        if subdir.startswith("_temp"):
            continue
            
        json_file_path = os.path.join(dataset_dir, subdir, f"{subdir}.json")
        
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_posts_data.append(data)
            except Exception as e:
                print(f"Errore nella lettura di {json_file_path}: {e}")
        else:
            # Fallback: search for any json file in the folder if the naming doesn't match perfectly
            found_json = False
            for file in os.listdir(os.path.join(dataset_dir, subdir)):
                if file.endswith(".json") and not any(x in file for x in ["_ocr", "_transcript", "_metadata"]):
                    try:
                        with open(os.path.join(dataset_dir, subdir, file), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_posts_data.append(data)
                            found_json = True
                            break
                    except:
                        continue
            if not found_json:
                print(f"Avviso: Nessun JSON trovato in {subdir}")

    if all_posts_data:
        output_file = os.path.join(dataset_dir, f"{profile_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_posts_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccesso! Creato file unico: {output_file}")
        print(f"Totale post inclusi: {len(all_posts_data)}")
    else:
        print("\nNessun dato trovato da combinare.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        profile = sys.argv[1]
    else:
        profile = input("Inserisci il nome del profilo Instagram: ").strip()
    
    if profile:
        combine_json_files(profile)
    else:
        print("Nome profilo non valido.")
