# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# def gerar_tsne(model, feature_type):
#     # 1. Carregar os dados gerados pelo seu pipeline
#     X_model = np.load(feature_type) # altere para o arquivo onde salvou os embeddings da rede
#     y = np.load("y.npy")

#     # Mapeamento de nomes para a legenda do gráfico
#     tech_names = ["gsm", "fm", "lte", "tetra", "dab", "dvbt"]

#     print("[*] Aplicando t-SNE... Isso pode levar alguns minutos.")
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
#     X_tsne = tsne.fit_transform(X_model)

#     # 2. Plotar o gráfico
#     plt.figure(figsize=(10, 8), dpi=300)
#     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

#     for idx, name in enumerate(tech_names):
#         indices = np.where(y == idx)
#         plt.scatter(
#             X_tsne[indices, 0], 
#             X_tsne[indices, 1], 
#             label=name, 
#             alpha=0.7, 
#             edgecolors="w", 
#             linewidth=0.5,
#             c=colors[idx]
#         )

#     plt.title("Visualização dos Embeddings Espaciais via t-SNE - "+model, fontsize=14, fontweight="bold")
#     plt.xlabel("Componente t-SNE 1", fontsize=14)
#     plt.ylabel("Componente t-SNE 2", fontsize=14)
#     plt.legend(title="Tecnologias", loc="best", fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("tsne_features.png")
#     plt.show()