import os
import pandas as pd


def gerar_resumo(results_dir):
    path_dados = os.path.join(results_dir, "dados_brutos.csv")
    df = pd.read_csv(path_dados)

    # ===== GERAR RESUMO =====

    resumo_data = []

    # balanceamento (tecnologia)
    tech_counts = df["tecnologia"].value_counts()
    for tech, count in tech_counts.items():
        resumo_data.append({
            "tipo": "tecnologia_count",
            "categoria": tech,
            "valor": count
        })

    # frequência
    freq_counts = df["freq"].value_counts()
    for freq, count in freq_counts.items():
        resumo_data.append({
            "tipo": "freq_count",
            "categoria": freq,
            "valor": count
        })

    # ruído por tecnologia
    std_mean = df.groupby("tecnologia")["std"].mean()
    for tech, val in std_mean.items():
        resumo_data.append({
            "tipo": "std_medio_tecnologia",
            "categoria": tech,
            "valor": val
        })
    # ===== ESTATÍSTICAS GLOBAIS =====

    resumo_data.append({
        "tipo": "global",
        "categoria": "total_arquivos",
        "valor": len(df)
    })

    resumo_data.append({
        "tipo": "global",
        "categoria": "mean_global",
        "valor": df["mean"].mean()
    })

    resumo_data.append({
        "tipo": "global",
        "categoria": "std_medio_global",
        "valor": df["std"].mean()
    })

    resumo_data.append({
        "tipo": "global",
        "categoria": "std_min",
        "valor": df["std"].min()
    })

    resumo_data.append({
        "tipo": "global",
        "categoria": "std_max",
        "valor": df["std"].max()
    })

    # ===== SALVAR CSV =====
    df_resumo = pd.DataFrame(resumo_data)

    df_resumo.to_csv(os.path.join(results_dir, "resumo_analise.csv"), index=False)

    print("CSV 'resumo_analise.csv' salvo em /results")