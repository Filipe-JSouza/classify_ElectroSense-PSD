import os
import numpy as np
from collections import Counter
import pandas as pd

# ===== PARSER ROBUSTO =====
def parse_filename(fname):
    name = os.path.splitext(fname)[0].lower()
    parts = name.split("_")

    tech = None
    freq = None
    location = None

    for i, p in enumerate(parts):
        if "spectrumbands" in p:
            try:
                # freq
                if parts[i+1].isdigit() and parts[i+2].isdigit():
                    freq = f"{parts[i+1]}_{parts[i+2]}"

                # tecnologia
                tech_candidate = parts[i+3]
                if tech_candidate.isalnum():
                    tech = tech_candidate

                # localização
                loc_candidate = parts[i+4]
                if loc_candidate.isalnum():
                    location = loc_candidate

            except IndexError:
                pass

    return tech, freq, location


def gerar_dados(ROOT_DIR, MAX_TOP_FOLDERS):

    # ===== LISTAR PASTAS =====
    top_folders = sorted([
        f for f in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, f))
    ])

    selected_folders = top_folders[:MAX_TOP_FOLDERS]

    # ===== ESTATÍSTICAS =====
    tech_counter = Counter()
    freq_counter = Counter()
    location_counter = Counter()
    shape_counter = Counter()

    means = []
    stds = []
    total_files = 0

    # ===== DADOS =====
    raw_data = []

    # ===== LISTA DE ERROS =====
    parsing_error_list = []

    # ===== LOOP PRINCIPAL =====
    for folder in selected_folders:
        folder_path = os.path.join(ROOT_DIR, folder)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".npy"):
                    full_path = os.path.join(root, file)

                    tech, freq, location = parse_filename(file)

                    # substitui None por "unknown"
                    tech = tech if tech else "unknown"
                    freq = freq if freq else "unknown"
                    location = location if location else "unknown"

                    # ===== detecta unknown =====
                    if "unknown" in [tech, freq, location]:
                        parsing_error_list.append(full_path)

                    try:
                        arr = np.load(full_path)

                        shape = arr.shape
                        mean_val = arr.mean()
                        std_val = arr.std()

                        shape_counter[shape] += 1
                        means.append(mean_val)
                        stds.append(std_val)

                        raw_data.append({
                            "tecnologia": tech,
                            "freq": freq,
                            "resolucao": str(shape),
                            "local": location,
                            "mean": mean_val,
                            "std": std_val
                        })

                        tech_counter[tech] += 1
                        freq_counter[freq] += 1
                        location_counter[location] += 1

                        total_files += 1

                    except Exception as e:
                        print(f"Erro ao ler: {file} -> {e}")

    # ===== SALVAR CSV =====
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    df_final = pd.DataFrame(raw_data)
    df_final.to_csv(os.path.join(results_dir, "dados_brutos.csv"), index=False)

    print("\nCSV 'dados_brutos.csv' salvo em /results")

    # ===== PRINT SIMPLES DOS ERROS =====
    # print("\n--- Arquivos com 'unknown' ---")
    # for caminho in parsing_error_list:
    #     print(caminho)

    return results_dir