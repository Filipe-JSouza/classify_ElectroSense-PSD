import os
import numpy as np
import matplotlib.pyplot as plt

def gerar_plot_antes_depois_estrito(ROOT_DIR):
    print("[*] Carregando as matrizes salvas pelo pipeline...")
    
    try:
        X_img = np.load(os.path.join(ROOT_DIR, "results", "X_img.npy"))
        y = np.load(os.path.join(ROOT_DIR, "results", "y.npy"))
        
        # Tenta carregar os paths salvos em formato numpy array
        if os.path.exists(os.path.join(ROOT_DIR, "results", "paths.npy")):
            paths = np.load(os.path.join(ROOT_DIR, "results", "paths.npy"), allow_pickle=True)
        else:
            # Caso não tenha o paths.npy, vamos tentar ler a variável original 
            # ou usar o arquivo 'groups.npy' combinado com o diretório padrão
            print("[Aviso] 'paths.npy' não encontrado. Usando fallback de busca por groups...")
            groups = np.load(os.path.join(ROOT_DIR, "results", "groups.npy"), allow_pickle=True)
            paths = None
    except FileNotFoundError as e:
        print(f"[Erro] Arquivos base não encontrados: {e}")
        print("Certifique-se de ter executado o pipeline principal e gerado os arquivos .npy no disco.")
        return

    # Dicionário oficial de mapeamento de IDs para Tecnologias
    IDX_TO_TECH = {0: "gsm", 1: "fm", 2: "lte", 3: "tetra", 4: "dab", 5: "dvbt"}
    
    # 1. Encontrar o índice da primeira ocorrência de cada uma das 6 tecnologias em 'y'
    amostras_alvo = {}
    for idx, label in enumerate(y):
        tech_name = IDX_TO_TECH[label]
        if tech_name not in amostras_alvo:
            amostras_alvo[tech_name] = idx
        if len(amostras_alvo) == 6:
            break

    # 2. Configurar a janela do Matplotlib (6 tecnologias x 2 colunas)
    fig, axes = plt.subplots(6, 2, figsize=(14, 20))
    fig.suptitle("Comparativo RF Estrito: Antes pré-processamento vs Depois pré-processamento", 
                 fontsize=16, fontweight='bold', y=0.96)

    # Ordenar as chaves para manter um padrão estético no plot
    tecnologias_ordenadas = sorted(amostras_alvo.keys())

    for idx, tech in enumerate(tecnologias_ordenadas):
        sample_idx = amostras_alvo[tech]
        
        # -----------------------------------------------------------------
        # PARTE 1: LEITURA DO ARQUIVO ORIGINAL (ANTES DO PREPROCESS) VIA PATH
        # -----------------------------------------------------------------
        psd_original = None
        
        if paths is not None:
            caminho_arquivo = paths[sample_idx]
            if os.path.exists(caminho_arquivo):
                psd_original = np.load(caminho_arquivo)
                nome_exibicao = os.path.basename(caminho_arquivo)
            else:
                print(f"[Aviso] O arquivo mapeado em path não foi localizado: {caminho_arquivo}")
        
        # Fallback de segurança caso o arquivo físico exato não esteja acessível no caminho gravado
        if psd_original is None:
            localidade = groups[sample_idx] if 'groups' in locals() else "Desconhecido"
            nome_exibicao = f"Amostra_{sample_idx}_{tech}_{localidade}.npy (Fallback)"
            # Como fallback visual para o plot não quebrar, recuperamos o canal do próprio X_img
            psd_original = X_img[sample_idx][0] 

        # -----------------------------------------------------------------
        # PARTE 2: RECUPERAÇÃO DA IMAGEM PROCESSADA (DEPOIS DO PREPROCESS)
        # -----------------------------------------------------------------
        img_processada = X_img[sample_idx]
        
        # Se a matriz está no formato PyTorch (3, 224, 224), transpõe para (224, 224, 3)
        if img_processada.ndim == 3 and img_processada.shape[0] == 3:
            img_processada = np.transpose(img_processada, (1, 2, 0))
            
        # Para a exibição no imshow com mapas de cores, extraímos o canal principal (caso tenha 3 canais)
        if img_processada.ndim == 3:
            img_display = img_processada[:, :, 0]
        else:
            img_display = img_processada

        # --- COLUNA 1: PLOT DO BRUTO ORIGINAL ---
        ax_antes = axes[idx, 0]
        im_antes = ax_antes.imshow(psd_original, aspect='auto', cmap='viridis')
        ax_antes.set_title(f"{tech.upper()} - Imagem Bruta", fontsize=10, fontweight='bold')
        fig.colorbar(im_antes, ax=ax_antes, fraction=0.046, pad=0.04)
        ax_antes.set_ylabel("Frequência (Bins)", fontsize=9)
        if idx == 5:
            ax_antes.set_xlabel("Eixo Temporal (Amostras)", fontsize=9)

        # --- COLUNA 2: PLOT DO X_IMG PROCESSADO ---
        ax_depois = axes[idx, 1]
        im_depois = ax_depois.imshow(img_display, aspect='auto', cmap='jet')
        ax_depois.set_title(f"{tech.upper()} - Imagem Pré-Processada", fontsize=10, fontweight='bold')
        fig.colorbar(im_depois, ax=ax_depois, fraction=0.046, pad=0.04)
        if idx == 5:
            ax_depois.set_xlabel("Largura Redimensionada (Pixels)", fontsize=9)

    # Ajustar o layout do gráfico para não sobrepor títulos e barras
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    nome_imagem = "resultado_comparativo_real.png"
    plt.savefig(nome_imagem, dpi=300)
    print(f"\n[✓] Gráfico comparativo gerado com sucesso! Arquivo salvo como: '{nome_imagem}'")
    plt.show()

    