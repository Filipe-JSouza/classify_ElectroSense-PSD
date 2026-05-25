# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

# def plot_all_8_curves(json_file, feature_type="CNN"):
#     """
#     Plota 8 curvas ROC no mesmo gráfico corrigindo o alinhamento 
#     multiclasse One-vs-Rest (OvR) fixado em 6 classes.
#     """
#     with open(json_file, "r") as f:
#         results = json.load(f)

#     # Filtrar estritamente pelo tipo de feature (ex: "CNN")
#     results = [r for r in results if r["FeatureType"] == feature_type]

#     # Lista de classificadores esperados
#     classifiers = ["SVM", "KNN", "LDA", "NB"]
    
#     # O seu problema possui exatamente 6 classes globais (0 a 5)
#     N_CLASSES_GLOBAL = 6
#     classes_globais = list(range(N_CLASSES_GLOBAL))

#     plt.figure(figsize=(12, 9))
#     mean_fpr = np.linspace(0, 1, 100)

#     # Cores fixas por classificador
#     colors = {
#         "SVM": "#1f77b4",  # Azul
#         "KNN": "#ff7f0e",  # Laranja
#         "LDA": "#2ca02c",  # Verde
#         "NB": "#d62728"    # Vermelho
#     }

#     for clf_name in classifiers:
#         subset = [r for r in results if r["Classifier"] == clf_name]
#         if not subset:
#             continue

#         # --- PARTE 1: PROCESSAR MÉDIA GERAL ---
#         all_tprs = []
#         all_aucs = []
#         hp_groups = {}

#         for r in subset:
#             y_true = np.array(r["y_true"])
#             y_prob = np.array(r["y_prob"])
            
#             # CORREÇÃO 1: Binarizar sempre usando o escopo total de 6 classes (evita quebra de colunas)
#             y_true_bin = label_binarize(y_true, classes=classes_globais)
            
#             # Caso o vetor de teste tenha apenas 1 classe (raro, mas possível), ajusta formato
#             if N_CLASSES_GLOBAL == 2 or y_true_bin.shape[1] == 1:
#                 y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

#             tpr_list_fold = []
#             auc_list_fold = []
            
#             # CORREÇÃO 2: Iterar estritamente pelas 6 colunas correspondentes de cada classe
#             for i in range(N_CLASSES_GLOBAL):
#                 # Verifica se a classe 'i' possui pelo menos um exemplo positivo e um negativo no fold
#                 if len(np.unique(y_true_bin[:, i])) == 2:
#                     fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#                     roc_auc_i = auc(fpr_i, tpr_i)
                    
#                     tpr_list_fold.append(np.interp(mean_fpr, fpr_i, tpr_i))
#                     auc_list_fold.append(roc_auc_i)
            
#             # Se o fold foi válido, calcula as médias macro deste fold
#             if tpr_list_fold:
#                 mean_tpr_fold = np.mean(tpr_list_fold, axis=0)
#                 mean_auc_fold = np.mean(auc_list_fold)
                
#                 all_tprs.append(mean_tpr_fold)
#                 all_aucs.append(mean_auc_fold)
#             else:
#                 # Fallback de segurança caso o fold seja muito desbalanceado
#                 mean_tpr_fold = np.zeros_like(mean_fpr)
#                 mean_auc_fold = 0.5

#             # --- IDENTIFICAR PARÂMETROS DO TUNING ---
#             if clf_name == "SVM":
#                 hp_key = f"C={r.get('C')}, G={r.get('Gamma')}"
#             elif clf_name == "KNN":
#                 hp_key = f"K={r.get('n_neighbors')}"
#             else:
#                 hp_key = "Padrão"

#             if hp_key not in hp_groups:
#                 hp_groups[hp_key] = {"tprs": [], "aucs": []}
            
#             hp_groups[hp_key]["tprs"].append(mean_tpr_fold)
#             hp_groups[hp_key]["aucs"].append(mean_auc_fold)

#         # 1. Plotar Curva Média Geral (Linha Contínua)
#         if all_tprs:
#             mean_tpr_geral = np.mean(all_tprs, axis=0)
#             mean_auc_geral = np.mean(all_aucs)
#             plt.plot(
#                 mean_fpr, 
#                 mean_tpr_geral, 
#                 color=colors[clf_name], 
#                 linestyle="-", 
#                 linewidth=2.5,
#                 label=f"{clf_name} Geral (Média AUC = {mean_auc_geral:.2f})"
#             )

#         # --- PARTE 2: ENCONTRAR E PLOTAR MELHOR TUNING ---
#         best_hp_key = None
#         best_hp_auc = -1
        
#         for hp_key, data in hp_groups.items():
#             avg_auc = np.mean(data["aucs"])
#             if avg_auc > best_hp_auc:
#                 best_hp_auc = avg_auc
#                 best_hp_key = hp_key

#         if best_hp_key:
#             best_tprs = hp_groups[best_hp_key]["tprs"]
#             mean_tpr_best_hp = np.mean(best_tprs, axis=0)
            
#             # 2. Plotar Curva do Melhor Tuning (Linha Tracejada)
#             plt.plot(
#                 mean_fpr, 
#                 mean_tpr_best_hp, 
#                 color=colors[clf_name], 
#                 linestyle="--", 
#                 linewidth=1.8,
#                 label=f"{clf_name} Melhor [{best_hp_key}] (AUC = {best_hp_auc:.2f})"
#             )

#     # Linha de referência da escolha aleatória (diagonal)
#     plt.plot([0, 1], [0, 1], linestyle=":", color="gray", alpha=0.7)
    
#     # Configurações do layout do gráfico
#     plt.xlabel("False Positive Rate (FPR)", fontsize=12)
#     plt.ylabel("True Positive Rate (TPR)", fontsize=12)
#     plt.title(f"Comparativo de Curvas ROC (Média Geral vs Melhor Tuning) - {feature_type}", fontsize=14, fontweight="bold")
#     plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
#     plt.grid(True, linestyle=":", alpha=0.5)
#     plt.xlim([-0.02, 1.02])
#     plt.ylim([-0.02, 1.02])
    
#     plt.tight_layout()
#     plt.show()