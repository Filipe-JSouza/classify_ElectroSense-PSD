import os
import src.analysis_dataset as ad
import src.analysis_dataresult as ar
import src.dataset_clean as dc
import src.preprocess as pp
import numpy as np
import src.classify as cl
import src.feature_extractors as fe
# from src.roc_curve import plot_all_8_curves 
# import src.tabelas as tb
# import src.t_sne as tsne
import src.img_plot as ip

# ===== CONFIG =====
ROOT_DIR= os.path.dirname(__file__)

DATA_DIR= os.path.join(ROOT_DIR, "data", "spectrum_bands_2")
DATA_DIR = os.path.abspath(DATA_DIR)
MAX_TOP_FOLDERS = 44
TRASH_DIR = os.path.join(ROOT_DIR, "data", "spectrum_bads_trash")
TRASH_DIR = os.path.abspath(TRASH_DIR)


if __name__ == "__main__":

    if not os.path.exists(TRASH_DIR) or len(os.listdir(TRASH_DIR)) == 0:
        #===== DADOS BRUTOS =====
        results_dir = ad.gerar_dados(DATA_DIR, MAX_TOP_FOLDERS)

        # ===== ANÁLISE DOS DADOS =====
        ar.gerar_resumo(results_dir)

        # ===== LIMPAR DATASET =====
        dc.limpar_dataset(DATA_DIR, TRASH_DIR, MAX_TOP_FOLDERS)

        # # 1. Forçar a leitura direta dos arquivos brutos (.npy originais das pastas)
        # print("[*] Lendo e aplicando pré-processamento do zero...")
        # X_img, X_extra, y, paths, groups = pp.read_data(DATA_DIR, MAX_TOP_FOLDERS)

        # # 2. Garantir formato correto de canais para o PyTorch ANTES de salvar
        # if X_img.shape[-1] == 3:  
        #     X_img = np.transpose(X_img, (0, 3, 1, 2))

        # # 3. Salvar os arquivos novos, agora perfeitamente alinhados em tamanho
        # print(f"[*] Salvando novos arquivos alinhados com {len(groups)} amostras...")
        # np.save("X_img.npy", X_img)
        # np.save("y.npy", y)
        # np.save("groups.npy", groups)
        # np.save("paths.npy", paths) 
        # print("[✓] Arquivos base salvos com sucesso!")
        
        # X_img = np.load(os.path.join(ROOT_DIR, "X_img.npy"))
        # y = np.load(os.path.join(ROOT_DIR, "y.npy"))
        # groups= np.load(os.path.join(ROOT_DIR, "groups.npy"))

        # unique_groups, counts = np.unique(groups, return_counts=True)
        # for g, c in zip(unique_groups, counts):
        #     print(f"Grupo (Localidade): {g} -> {c} amostras")

        # X_img = np.load("X_img.npy")
        # #X_extra = np.load("X_extra.npy")
        # y = np.load("y.npy")
        # groups= np.load("groups.npy")

        # # garantir formato correto
        # if X_img.shape[-1] == 3:  # caso esteja [N, 224, 224, 3]
        #   X_img = np.transpose(X_img, (0, 3, 1, 2))
    
        # #==== EXTRAIR FEATURES ViT =====
        # X_vit =fe.extract_vit_features(X_img)
        # #==== EXTRAIR FEATURES CNN =====
        # X_cnn =fe.extract_cnn_features(X_img)
    
        # #==== RODAR EXPERIMENTOS DE CLASSIFICAÇÃO =====
        # results = cl.run_experiments(X_cnn, X_vit, y, groups=groups)

        # #==== GERAR CURVAS ROC =====
        # plot_all_8_curves("results.json", feature_type="CNN")

        # #==== GERAR TABELAS DE RESULTADOS =====
        # tb.extrair_e_gerar_tabelas("results.json")

        # #==== GERAR T-SNE =====
        # tsne.gerar_tsne("CNN", "X_cnn_features.npy")
        # tsne.gerar_tsne("ViT", "X_vit_features.npy")

    #==== GERAR GRÁFICO COMPARATIVO ANTES VS DEPOIS DO PREPROCESS =====
    ip.gerar_plot_antes_depois_estrito(ROOT_DIR)

