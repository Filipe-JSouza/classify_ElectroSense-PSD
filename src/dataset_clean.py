import os
import shutil
import numpy as np


def limpar_dataset(ROOT_DIR, TRASH_DIR, MAX_TOP_FOLDERS=None):
    
    os.makedirs(TRASH_DIR, exist_ok=True)

    # listar pastas
    top_folders = sorted([
        f for f in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, f))
    ])

    if MAX_TOP_FOLDERS:
        top_folders = top_folders[:MAX_TOP_FOLDERS]

    total = 0
    movidos = 0

    for folder in top_folders:
        folder_path = os.path.join(ROOT_DIR, folder)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".npy"):
                    full_path = os.path.join(root, file)

                    mover = False

                    try:
                        arr = np.load(full_path)

                        # ===== REGRA 1: vazio ou shape inválido =====
                        if arr is None or arr.size == 0 or len(arr.shape) < 2:
                            mover = True

                        if arr.ndim < 2:
                            mover = True

                        if 0 in arr.shape:
                            mover = True

                        # ===== REGRA 2: muito pequeno =====
                        if arr.shape[0] < 10 or arr.shape[1] < 10:
                            mover = True

                        # ===== REGRA 3: muito grande (outlier) =====
                        if arr.shape[1] > 10000:
                            mover = True

                        # ===== REGRA 4: NaN =====
                        if np.isnan(arr).any():
                            mover = True

                        # ===== REGRA 5: estatísticas inválidas =====
                        mean_val = np.mean(arr)
                        std_val = np.std(arr)

                        if np.isnan(mean_val) or np.isnan(std_val):
                            mover = True

                        # ===== REGRA 6: valores absurdos =====
                        if std_val < 1 or std_val > 20:
                            mover = True

                    except Exception:
                        mover = True

                    # ===== mover =====
                    if mover:
                        destino = os.path.join(TRASH_DIR, file)

                        # evitar sobrescrever
                        if os.path.exists(destino):
                            base, ext = os.path.splitext(file)
                            destino = os.path.join(TRASH_DIR, f"{base}_{movidos}{ext}")

                        shutil.move(full_path, destino)
                        movidos += 1

                    total += 1

    print("\n--- Limpeza concluída ---")
    print(f"Arquivos analisados: {total}")
    print(f"Arquivos movidos: {movidos}")