import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from tqdm import tqdm
import ollama

# =====================================================================
# 1. CONFIGURAZIONE E SETUP INIZIALE
# =====================================================================
SOURCES = [
    'data/istat/AVQ_Microdati_2023.txt',
    'data/istat/AVQ_Microdati_2022.txt',
    'data/istat/AVQ_Microdati_2021.txt'
]
OUTPUT_JSON = 'data/processed/ground_truth_multiyr.json'
LLM_MODEL = 'qwen3:14b'

TARGET_COLS = ['ETAMi', 'COEFIN', 'CONDMi', 'ISTRMi', 'CAMCLI', 'SITEC', 'PUNTIFI5', 'ANNO', 'IMPITA7', 'SICURO']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 2. CARICAMENTO DATI
# =====================================================================
def load_all_sources(file_list):
    combined_df = []
    for file in file_list:
        if os.path.exists(file):
            logging.info(f"Lettura file in corso: {file}")
            df = pd.read_csv(file, sep='\t', dtype=str, usecols=lambda c: c.strip() in TARGET_COLS)
            df.columns = df.columns.str.strip()
            
            for col in TARGET_COLS:
                if col not in df.columns:
                    df[col] = np.nan
            combined_df.append(df)
        else:
            logging.warning(f"File saltato (non trovato nel percorso): {file}")
    
    if not combined_df:
        return pd.DataFrame()
    return pd.concat(combined_df, ignore_index=True)

# =====================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =====================================================================
def process_data(df):
    logging.info("Pulizia e Feature Engineering in corso...")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            
    giovani_codici = ['007', '008', '009']
    df_y = df[df['ETAMi'].isin(giovani_codici)].copy()
    
    df_y['COEFIN'] = pd.to_numeric(df_y['COEFIN'], errors='coerce').fillna(1)
    df_y['COEFIN'] = (df_y['COEFIN'] / 10000.0) / len(SOURCES)
    
    cond_map = {'1': 'occupato', '2': 'disoccupato', '3': 'studente o inattivo'}
    df_y['lavoro'] = df_y['CONDMi'].map(cond_map).fillna('non specificato')
    
    df_y['clima_top'] = df_y['CAMCLI'] == '03'
    df_y['insoddisfatto_econ'] = df_y['SITEC'].isin(['3', '4'])
    
    df_y['PUNTIFI5'] = df_y['PUNTIFI5'].replace('', np.nan)
    df_y['sfiducia_politica'] = pd.to_numeric(df_y['PUNTIFI5'], errors='coerce') <= 3
    
    df_y['paura_sicurezza'] = df_y['SICURO'].isin(['3', '4'])
    
    df_y['IMPITA7'] = df_y['IMPITA7'].replace('', np.nan)
    df_y['pro_diritti_civili'] = df_y['IMPITA7'].isin(['1', '2'])
    
    return df_y

# =====================================================================
# 4. REPORTISTICA
# =====================================================================
def print_dataset_statistics(df_youth):
    youth_pop = df_youth['COEFIN'].sum()

    print("\n" + "="*65)
    print("📊 REPORT STATISTICO POPOLAZIONE (Base 6 Dimensioni)")
    print("="*65)
    print(f"Record effettivi intervistati (18-34) : {len(df_youth):,}".replace(',', '.'))
    print(f"Popolazione giovanile reale stimata   : {int(youth_pop):,}".replace(',', '.'))
    print("-" * 65)
    
    def print_stat(col_name, true_label, false_label):
        stats = df_youth.groupby(col_name)['COEFIN'].sum()
        for idx, val in stats.items():
            label = true_label if idx else false_label
            pct = (val / youth_pop) * 100
            print(f"  - {label.ljust(35)}: {pct:>5.1f}%")
        print("-" * 65)

    print_stat('clima_top', "Preoccupati per l'Eco-crisi", "Clima non è priorità assoluta")
    print_stat('pro_diritti_civili', "Diritti Civili (Priorità Alta)", "Diritti Civili (Priorità Bassa)")
    print_stat('insoddisfatto_econ', "Insoddisfatto situazione economica", "Economicamente stabile/sereno")
    print_stat('paura_sicurezza', "Insicuro/Paura della criminalità", "Si sente sicuro in strada")
    print_stat('sfiducia_politica', "Sfiduciato dalla Politica (<= 3/10)", "Fiducia Politica nella media")
    print("="*65 + "\n")

# =====================================================================
# 5. GENERAZIONE GROUND TRUTH (CON SOGLIA CUMULATIVA E CHECKPOINT)
# =====================================================================
def generate_ground_truth(df, coverage_threshold=0.90):
    group_cols = ['lavoro', 'clima_top', 'insoddisfatto_econ', 'sfiducia_politica', 'pro_diritti_civili', 'paura_sicurezza']
    
    # Raggruppamento sicuro con dropna=False
    archetypes = df.groupby(group_cols, dropna=False)['COEFIN'].sum().reset_index()
    archetypes = archetypes.sort_values('COEFIN', ascending=False)
    
    # Logica di Copertura Cumulativa (Metodo Scientifico)
    total_weight = archetypes['COEFIN'].sum()
    archetypes['weight_pct'] = (archetypes['COEFIN'] / total_weight) * 100
    archetypes['cum_pct'] = archetypes['COEFIN'].cumsum() / total_weight
    
    # Filtriamo per coprire la soglia richiesta (es. 90%)
    filtered_archetypes = archetypes[archetypes['cum_pct'] <= coverage_threshold].copy()
    
    num_clusters = len(filtered_archetypes)
    logging.info(f"Selezionati {num_clusters} archetipi per coprire il {coverage_threshold*100}% della popolazione.")
    
    results = []
    
    for i, row in tqdm(filtered_archetypes.iterrows(), total=num_clusters, desc="Inferenza LLM"):
        context = (
            f"Un giovane italiano {row['lavoro']}. "
            f"{'È fortemente preoccupato per il cambiamento climatico. ' if row['clima_top'] else 'Non reputa l\'ambiente una priorità urgente. '}"
            f"{'Vive una situazione di insoddisfazione economica. ' if row['insoddisfatto_econ'] else 'È economicamente sereno. '}"
            f"{'Ritiene i diritti civili e l\'inclusione una priorità assoluta. ' if row['pro_diritti_civili'] else 'Non ritiene i diritti civili una priorità urgente. '}"
            f"{'Teme la criminalità urbana e si sente insicuro in strada. ' if row['paura_sicurezza'] else 'Si sente sicuro nella sua città. '}"
            f"{'Prova totale sfiducia verso i partiti politici.' if row['sfiducia_politica'] else 'Ha un livello normale di fiducia nelle istituzioni.'}"
        )
        
        # PROMPT DE-BIASED: Rimossi i trigger per lo slang caricaturale
        prompt = f"""
        Sei un data scientist. Il tuo compito è dare voce a un cluster statistico (Persona).
        Profilo assegnato: {context}
        
        Scrivi ESATTAMENTE 3 brevi pensieri (massimo 15 parole ciascuno) che questa specifica persona scriverebbe sui social media (es. X/Twitter) per esprimere le sue frustrazioni o priorità.
        
        REGOLE FONDAMENTALI:
        - Usa un registro colloquiale ma naturale e autentico.
        - EVITA ASSOLUTAMENTE lo slang forzato, gli stereotipi giovanili (niente "raga", "bro", "cringe") e i toni caricaturali.
        - Niente hashtag o emoji.
        - Sii incisivo e realistico.
        
        Separa i tre commenti ESATTAMENTE con questo simbolo: |
        """
        
        try:
            response = ollama.generate(model=LLM_MODEL, prompt=prompt)
            lines = [l.strip() for l in response['response'].split('|') if len(l.strip()) > 5]
            
            results.append({
                "weight_pct": round(row['weight_pct'], 3),
                "represented_youth": int(row['COEFIN']),
                "profile_description": context.strip(),
                "generated_statements": lines[:3]
            })
            
            # CHECKPOINT: Salvataggio continuo ad ogni step per evitare perdite di dati
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Errore durante l'inferenza LLM al cluster {i}: {e}")
            
    return results

# =====================================================================
# 6. ESECUZIONE MAIN SCRIPT
# =====================================================================
if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    logging.info("Inizio esecuzione pipeline Human Data Science...")
    
    raw_data = load_all_sources(SOURCES)
    if raw_data.empty:
        logging.error("Nessun dato caricato. Controlla i percorsi e i nomi dei file.")
        sys.exit(1)
        
    youth_data = process_data(raw_data)
    print_dataset_statistics(youth_data)
    
    user_input = input(f"Logica aggiornata (Soglia 90% & Checkpointing). Avviare l'inferenza {LLM_MODEL}? [y/N]: ")
    
    if user_input.strip().lower() != 'y':
        logging.info("Esecuzione interrotta. Nessun dato generato.")
        sys.exit(0)
        
    # Copriamo il 90% della demografia (puoi alzare a 0.95 se vuoi più archetipi)
    generate_ground_truth(youth_data, coverage_threshold=0.90)
    
    logging.info(f"✅ FASE A COMPLETATA! Ground Truth salvata in: {OUTPUT_JSON}")