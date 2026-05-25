# import numpy as np
# import pandas as pd
# import json
# from sklearn.model_selection import GroupKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.metrics import accuracy_score
# from sklearn.dummy import DummyClassifier

# def run_experiments(X_cnn, X_vit, y, groups, out_json="results.json", out_csv="results.csv"):
#     gkf = GroupKFold(n_splits=3)
#     results = []

#     C_values = [0.1, 1, 10, 100]
#     gamma_values = ["scale", "auto", 0.001, 0.01, 0.1]
#     k_values = [1, 3, 5, 7, 9]

#     for feature_type, X in [("CNN", X_cnn), ("ViT", X_vit)]:
#         for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]

#             # --- O PULO DO GATO: Descobrir quais locais únicos estão neste teste ---
#             locais_teste = np.unique(groups[val_idx])
#             # Une os nomes com hífen caso haja mais de um local no mesmo fold (ex: "bu-scalessio")
#             nome_local_teste = "-".join(locais_teste) 

#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_val = scaler.transform(X_val)

#             # --- 1. SVM TUNING ---
#             for C in C_values:
#                 for gamma in gamma_values:
#                     clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)
#                     clf.fit(X_train, y_train)
#                     y_pred = clf.predict(X_val)
#                     y_prob = clf.predict_proba(X_val) 
#                     acc = accuracy_score(y_val, y_pred)

#                     results.append({
#                         "FeatureType": feature_type,
#                         "Classifier": "SVM",
#                         "Fold": fold_idx,
#                         "TestLocation": nome_local_teste, # <--- SALVA O NOME DO LOCAL AQUI!
#                         "C": C,
#                         "Gamma": str(gamma) if isinstance(gamma, str) else gamma,
#                         "Accuracy": acc,
#                         "y_true": y_val.tolist(),
#                         "y_prob": y_prob.tolist()
#                     })

#             # --- 2. KNN TUNING ---
#             for k in k_values:
#                 clf = KNeighborsClassifier(n_neighbors=k)
#                 clf.fit(X_train, y_train)
#                 y_pred = clf.predict(X_val)
#                 y_prob = clf.predict_proba(X_val) 
#                 acc = accuracy_score(y_val, y_pred)
                
#                 results.append({
#                     "FeatureType": feature_type,
#                     "Classifier": "KNN",
#                     "Fold": fold_idx,
#                     "TestLocation": nome_local_teste, # <--- SALVA O NOME DO LOCAL AQUI!
#                     "n_neighbors": k,
#                     "Accuracy": acc,
#                     "y_true": y_val.tolist(),
#                     "y_prob": y_prob.tolist()
#                 })

#             # --- 3. LDA ---
#             clf = LinearDiscriminantAnalysis()
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_val)
#             y_prob = clf.predict_proba(X_val) 
#             acc = accuracy_score(y_val, y_pred)
            
#             results.append({
#                 "FeatureType": feature_type,
#                 "Classifier": "LDA",
#                 "Fold": fold_idx,
#                 "TestLocation": nome_local_teste, # <--- SALVA O NOME DO LOCAL AQUI!
#                 "Accuracy": acc,
#                 "y_true": y_val.tolist(),
#                 "y_prob": y_prob.tolist()
#             })

#             # --- 4. NAIVE BAYES ---
#             clf = GaussianNB()
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_val)
#             y_prob = clf.predict_proba(X_val) 
#             acc = accuracy_score(y_val, y_pred)
            
#             results.append({
#                 "FeatureType": feature_type,
#                 "Classifier": "NB",
#                 "Fold": fold_idx,
#                 "TestLocation": nome_local_teste, # <--- SALVA O NOME DO LOCAL AQUI!
#                 "Accuracy": acc,
#                 "y_true": y_val.tolist(),
#                 "y_prob": y_prob.tolist()
#             })
#             # --- 5. DUMMY CLASSIFIER (BASELINE DE SANIDADE) ---
#             # Ele vai apenas chutar a classe mais frequente do treino
#             clf_dummy = DummyClassifier(strategy="most_frequent")
#             clf_dummy.fit(X_train, y_train)
#             y_pred_dummy = clf_dummy.predict(X_val)
#             y_prob_dummy = clf_dummy.predict_proba(X_val) 
#             acc_dummy = accuracy_score(y_val, y_pred_dummy)
            
#             results.append({
#                 "FeatureType": feature_type,
#                 "Classifier": "Dummy",
#                 "Fold": fold_idx,
#                 "TestLocation": nome_local_teste,
#                 "Accuracy": acc_dummy,
#                 "y_true": y_val.tolist(),
#                 "y_prob": y_prob_dummy.tolist()
#             })

#     with open(out_json, "w") as f:
#         json.dump(results, f, indent=2)

#     df = pd.DataFrame(results)
#     df.to_csv(out_csv, index=False)
#     return results