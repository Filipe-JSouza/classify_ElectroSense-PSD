import os
import numpy as np
import cv2

# =========================
# CONFIG
# =========================
TARGET_WIDTH = 1024
IMG_SIZE = (224, 224) #(224, 128)

VALID_TECHS = {"gsm", "fm", "lte", "tetra", "dab", "dvbt"}


# =========================
# PARSER (SEU CÓDIGO)
# =========================
def parse_filename(fname):
    name = os.path.splitext(fname)[0].lower()
    parts = name.split("_")

    tech = None
    freq = None
    location = None

    for i, p in enumerate(parts):
        if "spectrumbands" in p:
            try:
                if parts[i+1].isdigit() and parts[i+2].isdigit():
                    freq = f"{parts[i+1]}_{parts[i+2]}"

                tech_candidate = parts[i+3]
                if tech_candidate.isalnum():
                    tech = tech_candidate

                loc_candidate = parts[i+4]
                if loc_candidate.isalnum():
                    location = loc_candidate

            except IndexError:
                pass
    if location == "932":
        location = "fr"
    return tech, freq, location


# =========================
# DOWNSAMPLE
# =========================
def downsample_width(psd, target_width):
    h, w = psd.shape

    if w <= target_width:
        return psd

    factor = w // target_width
    psd = psd[:, :factor * target_width]
    psd = psd.reshape(h, target_width, factor)
    psd = psd.mean(axis=2)

    return psd


# =========================
# PREPROCESS
# =========================
def preprocess(psd):

    psd = psd.astype(np.float32)

    # 1. clipping baseado nos dados
    psd = np.clip(psd, -80, -20)

    # 2. normalização por amostra
    mean = psd.mean()
    std = psd.std()
    if std < 1e-6:
        std = 1e-6

    psd = (psd - mean) / std

    # 3. limitar extremos
    psd = np.clip(psd, -3, 3)

    # 4. downsample
    psd = downsample_width(psd, TARGET_WIDTH)

    # 5. resize
    psd = cv2.resize(psd, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 6. normalizar para [0,1]
    psd = (psd - psd.min()) / (psd.max() - psd.min() + 1e-8)

    # 7. 3 canais
    psd_img = np.stack([psd]*3, axis=0)

    return psd_img.astype(np.float32)


# =========================
# EXTRA FEATURES
# =========================
def extract_extra_features(psd, freq):

    features = []

    # estatísticas básicas
    features.append(psd.mean())
    features.append(psd.std())
    features.append(psd.max())
    features.append(psd.min())

    # largura original (importante!)
    features.append(psd.shape[1])

    # faixa de frequência
    if freq is not None:
        try:
            fmin, fmax = map(float, freq.split("_"))
            features.append(fmax - fmin)
            features.append((fmax + fmin) / 2.0)
        except:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


# =========================
# LABEL ENCODING
# =========================
TECH_TO_IDX = {
    "gsm": 0,
    "fm": 1,
    "lte": 2,
    "tetra": 3,
    "dab": 4,
    "dvbt": 5
}


# =========================
# LOAD DATASET
# =========================
def read_data(ROOT_DIR, MAX_TOP_FOLDERS):

    X_img = []
    X_extra = []
    y = []
    paths = []
    groups = []  

    # listar pastas
    top_folders = sorted([
        f for f in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, f))
    ])

    selected_folders = top_folders[:MAX_TOP_FOLDERS]

    for folder in selected_folders:
        folder_path = os.path.join(ROOT_DIR, folder)

        for root, dirs, files in os.walk(folder_path):
            for file in files:

                if not file.endswith(".npy"):
                    continue

                full_path = os.path.join(root, file)

                # parse filename
                tech, freq, location = parse_filename(file)

                # filtrar techs válidas
                if tech not in VALID_TECHS:
                    continue

                try:
                    psd = np.load(full_path)

                    if psd.ndim != 2:
                        continue

                    # preprocess imagem
                    psd_img = preprocess(psd)

                    # features extras (usam psd original!)
                    extra_feat = extract_extra_features(psd, freq)

                    # label
                    label = TECH_TO_IDX[tech]

                    # salvar
                    X_img.append(psd_img)
                    X_extra.append(extra_feat)
                    y.append(label)
                    paths.append(full_path)
                    groups.append(location)

                except Exception as e:
                    print(f"Erro em {full_path}: {e}")

    # converter para numpy
    X_img = np.array(X_img, dtype=np.float32)
    X_extra = np.array(X_extra, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)

    return X_img, X_extra, y, paths, groups
