# Classificação de Sinais de Radiofrequência (RF) com Deep Learning

Este repositório contém a pipeline completa de Engenharia de Dados, Extração de Características e Aprendizado de Máquina para a identificação automatizada de tecnologias de comunicação sem fios (RF) a partir de espectrogramas e Densidades Espectrais de Potência (PSD).

O projeto realiza a classificação de **6 tecnologias de telecomunicações** em cenários de validação cruzada entre ambientes (*Cross-Environment Validation*), utilizando representações visuais processadas por Redes Convolucionais (**ResNet50**) e Transformadores de Visão (**ViT**).

---

## 🏗️ Pipeline do Projeto

O fluxo de execução do projeto está estruturado em módulos independentes e sequenciais, projetados para garantir reprodutibilidade e isolamento estatístico:
[Dados Brutos .npy] ──> [1. Pré-processamento] ──> [2. Extração de Features] ──> [3. Classificação Cruzada]
(Zenodo Dataset)        - Clipping (-80 dB)        - CNN (ResNet50) Embeddings     - SVM / KNN / NB / LDA
- Normalização Z-Score     - ViT (Vision Transformer)      - GroupKFold (Por Local)
- Redimensionar (224x224)

1. **Pré-processamento (`src/preprocess.py`)**:
   * Realiza o *parsing* estrito dos nomes dos ficheiros brutos para extrair metadados (Tecnologia, Frequência, Localidade).
   * Aplica filtros de teto/piso dinâmicos (clipping em `-80 dB`) para eliminação do ruído de fundo do espectro.
   * Executa a normalização estatística *Z-Score* global para uniformização da amplitude do sinal.
   * Ajusta a geometria assimétrica dos ficheiros brutos convertendo-os em matrizes simétricas padronizadas de `224x224` pixels com 3 canais de cor expandidos para compatibilidade com arquiteturas de Visão Computacional.
   * **Tratamento de Anomalias de Dados**: O módulo possui tratamento automatizado para corrigir inconsistências de rotulação de locais geográficos na origem (ex: conversão da string espúria `932` para a classe de localidade válida `fr`).

2. **Extração de Características de Deep Learning**:
   * **CNN Features**: Passa as imagens tratadas pela camada de pooling global de uma ResNet50 pré-treinada para extrair vetores de características profundas de alta densidade discriminatória.
   * **ViT Features**: Codifica os padrões espaciais e dependências de longo alcance dos patches espectrais utilizando as características extraídas de um Vision Transformer.

3. **Treino e Classificação Cruzada (`src/classify.py` / `main.py`)**:
   * O ecossistema avalia o poder preditivo usando múltiplos classificadores clássicos: *Support Vector Machines (SVM)*, *K-Nearest Neighbors (KNN)*, *Linear Discriminant Analysis (LDA)* e *Naive Bayes (NB)*.
   * Garante a robustez através da validação **GroupKFold (3 Splits)** baseada estritamente na localidade geográfica. Isso assegura que o modelo nunca teste dados pertencentes a um ambiente que ele viu no treino, avaliando a capacidade real de generalização do modelo no mundo real.

---

## 📊 Dataset Utilizado

O projeto utiliza o dataset aberto de espectros de Radiofrequência disponível e hospedado no **Zenodo**. 

* **Referência de Download**: [https://zenodo.org/records/7521246](https://zenodo.org/records/7521246)
* **Classes Alvo (6 Tecnologias)**: `fm`, `gsm`, `lte`, `tetra`, `dab`, `dvbt`.
* **Volume do Dataset Limpo**: 225 amostras tridimensionais alinhadas pós-limpeza.
* **Mapeamento de Validação Cruzada (GroupKFold)**:
  Com base no perfil geográfico de captura coletado das amostras, as partições dos Folds distribuem-se estritamente sem vazamento de dados (*Data Leakage*) nos seguintes grupos de teste:
  * **Fold 0 (Grupo de Teste)**: `bu`, `cz`, `esp`, `fr`, `ge`, `ita`, `oh`, `rus`
  * **Fold 1 (Grupo de Teste)**: `alcorcon`, `jap`, `ne`, `scalessio`, `slv`, `sw`, `switz`, `test`, `usa`
  * **Fold 2 (Grupo de Teste)**: `cro`, `czk`, `ger`, `no`, `po`, `pol`, `rack3`, `swis`

---

## 📂 Estrutura de Pastas Obrigatória

Para que a pipeline execute com sucesso sem erros de indexação ou caminhos quebrados, o seu diretório de trabalho deve respeitar a árvore estrutural descrita abaixo. Você **deve criar** manualmente as pastas `data/` e `results/` caso elas não existam:

📂 seu-repositorio/
├── 📂 data/                         <-- VOCÊ DEVE CRIAR ESTA PASTA
│   └── 📂 spectrum_bands_2/         <-- Conteúdo extraído direto do arquivo ZIP do Zenodo
│       ├── 📄 spectrumBands_10_20_fm_alcorcon.npy
│       ├── 📄 spectrumBands_30_40_gsm_bu.npy
│       └── ... (demais arquivos brutas .npy baixados)
├── 📂 results/                      <-- VOCÊ DEVE CRIAR ESTA PASTA
│   └── 📄 results.json              <-- Gerado automaticamente pelo pipeline com predições e métricas
### Como configurar as dependências de arquivos:

1. **Pasta `data/`**: 
   * Crie uma pasta chamada `data` na raiz do projeto.
   * Baixe o arquivo compactado do link do Zenodo.
   * Extraia o seu conteúdo diretamente dentro de `data/`. Certifique-se de que a subpasta `spectrum_bands_2` contendo centenas de arquivos brutos `.npy` está diretamente sob o diretório `data/`.

2. **Pasta `results/`**:
   * Crie uma pasta chamada `results` na raiz do projeto.
   * Esta pasta será o alvo de escrita do pipeline de experimentos. O arquivo `results.json` será depositado nela contendo as probabilidades preditas de cada fold (`y_prob`), os rótulos de teste reais (`y_true`), acurácias obtidas e os parâmetros de busca exatos.

---

## 🚀 Como Executar o Projeto

1. Certifique-se de preencher a pasta `data/` conforme as instruções acima.
2. Instale as dependências requeridas pelo ecossistema:
   ```bash
   pip install numpy pandas scikit-learn opencv-python matplotlib tabulate

Execute o script principal para rodar a extração e os classificadores:
   python main.py