# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

# def extrair_e_gerar_tabelas(json_file="results.json"):
#     print(f"[*] Lendo dados de {json_file}...")
#     with open(json_file, "r") as f:
#         data = json.load(f)

#     # 1. Mapeamento estrito dos Folds para os Grupos Reais do seu dataset
#     # (Baseado nas partições geradas pelo GroupKFold no seu conjunto de dados)
#     fold_to_locations = {
#         0: "bu, esp, fr, ge, oh, po, sw, switz",
#         1: "932, alcorcon, cz, ita, ne, pol, scalessio, slv, usa",
#         2: "cro, czk, ger, jap, no, rack3, rus, swis, test"
#     }

#     N_CLASSES_GLOBAL = 6
#     classes_globais = list(range(N_CLASSES_GLOBAL))
    
#     rows = []

#     # 2. Processar cada registro do JSON e calcular a AUC-ROC de forma consistente
#     for r in data:
#         f_type = r["FeatureType"]
#         clf = r["Classifier"]
#         fold = r["Fold"]
#         acc = r["Accuracy"]
#         y_true = np.array(r["y_true"])
#         y_prob = np.array(r["y_prob"])
        
#         # Identificar e normalizar os nomes dos hiperparâmetros aplicados
#         if clf == "SVM":
#             hp = f"C={r.get('C')}, G={r.get('Gamma')}"
#         elif clf == "KNN":
#             hp = f"K={r.get('n_neighbors')}"
#         else:
#             hp = "Padrão"
            
#         # Binarização robusta para cálculo One-vs-Rest (OvR)
#         y_true_bin = label_binarize(y_true, classes=classes_globais)
        
#         auc_list_fold = []
#         for i in range(N_CLASSES_GLOBAL):
#             if len(np.unique(y_true_bin[:, i])) == 2:
#                 fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#                 auc_list_fold.append(auc(fpr_i, tpr_i))
        
#         # Média macro das AUCs das classes presentes no teste
#         mean_auc = np.mean(auc_list_fold) if auc_list_fold else 0.5
        
#         rows.append({
#             "FeatureType": f_type,
#             "Classifier": clf,
#             "Fold": fold,
#             "LocalTeste": fold_to_locations.get(fold, f"Fold {fold}"),
#             "Hiperparam": hp,
#             "Accuracy": acc,
#             "AUC": mean_auc
#         })

#     df_flat = pd.DataFrame(rows)

#     # =========================================================================
#     # TABELA 1: MELHORES MODELOS OTIMIZADOS (MÉDIA GERAL ENTRE FOLDS)
#     # =========================================================================
#     print("\n" + "="*80)
#     print(" TABELA 1: COMPARAÇÃO GERAL DOS MODELOS OTIMIZADOS (MÉDIA DOS FOLDS) ")
#     print("="*80)
    
#     # Agrupa por configuração para tirar a média dos folds
#     df_grouped = df_flat.groupby(["FeatureType", "Classifier", "Hiperparam"])[["Accuracy", "AUC"]].mean().reset_index()
    
#     # Identifica o melhor hiperparâmetro de cada classificador baseado na maior AUC média
#     idx_best = df_grouped.groupby(["FeatureType", "Classifier"])["AUC"].idxmax()
#     df_best_models = df_grouped.loc[idx_best].sort_values(by=["FeatureType", "AUC"], ascending=[True, False])
    
#     # Exibe em formato de tabela Markdown estruturada
#     print(df_best_models.to_markdown(index=False, floatfmt=".4f"))

#     # =========================================================================
#     # TABELA 2: ANÁLISE DETALHADA CROSS-ENVIRONMENT (FOCO EM LDA E NB)
#     # =========================================================================
#     print("\n" + "="*80)
#     print(" TABELA 2: COMPORTAMENTO DETALHADO POR GRUPO DE TESTE (LDA E NAIVE BAYES) ")
#     print("="*80)
    
#     # Filtra pelos modelos de destaque estatístico generativo para ver as flutuações por local
#     df_details = df_flat[df_flat["Classifier"].isin(["NB", "LDA"])].sort_values(by=["FeatureType", "Classifier", "Fold"])
    
#     # Reorganiza as colunas para melhor legibilidade no relatório técnico
#     cols_order = ["FeatureType", "Classifier", "Fold", "LocalTeste", "Accuracy", "AUC"]
#     df_details_print = df_details[cols_order]
    
#     print(df_details_print.to_markdown(index=False, floatfmt=".4f"))
#     print("="*80 + "\n")
    
#     # Opcional: Exportar os resumos consolidados para novos arquivos CSV limpos
#     df_best_models.to_csv("tcc_melhores_modelos.csv", index=False)
#     df_details_print.to_csv("tcc_detalhes_ambientes.csv", index=False)
#     print("[*] Arquivos 'tcc_melhores_modelos.csv' e 'tcc_detalhes_ambientes.csv' gerados!")
