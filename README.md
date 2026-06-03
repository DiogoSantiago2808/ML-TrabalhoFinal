# Clusterização de Clientes E-Commerce (Olist Public Dataset)

Este projeto tem como objetivo realizar a segmentação de clientes da plataforma de e-commerce brasileira **Olist**. Utilizando técnicas de Machine Learning não supervisionado (**K-Means** e **GMM - Gaussian Mixture Models**), o pipeline identifica perfis de comportamento de compra para direcionar estratégias de marketing, retenção (churn) e Growth.

Para lidar com a alta dimensionalidade e variáveis correlacionadas, o projeto aplica a Análise de Componentes Principais (**PCA**) após o tratamento estatístico e padronização dos dados brutos.

## 📁 Estrutura do Projeto

```text
meu_projeto_olist/
│
├── dados/                           # CSVs originais do Kaggle (Olist)
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   └── ...
│
├── src/                             # Módulos Python (Lógica do negócio)
│   ├── __init__.py                  # Inicializador do pacote
│   ├── processamento.py             # Carga, merge, tratamento de log e escala
│   ├── dimensionalidade.py          # Ajuste e análise de variância do PCA
│   └── modelagem.py                 # Algoritmos de clusterização e métricas
│
├── clusterizacao_clientes.ipynb     # Notebook principal de execução
└── README.md                        # Documentação do projeto
```

## 🛠️ Tecnologias e Dependências

As seguintes ferramentas foram utilizadas no desenvolvimento deste ecossistema:

- **Python 3.x**
- **Pandas & NumPy:** Manipulação e consolidação matricial dos dados.
- **Scikit-Learn:** Engenharia de atributos (`StandardScaler`), Redução de dimensionalidade (`PCA`) e Algoritmos de Clusterização (`KMeans`, `GaussianMixture`).
- **Matplotlib & Seaborn:** Visualização de dados e plotagem de curvas de validação.

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório e organizar os dados

Certifique-se de baixar o conjunto de dados [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) no Kaggle e extrair todos os arquivos `.csv` dentro do diretório `dados/`.

### 2. Instalar as dependências

No seu terminal, instale os pacotes necessários:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook

```

### 3. Rodar o Pipeline pelo Jupyter Notebook

Abra o arquivo `clusterizacao_clientes.ipynb`. Graças à configuração de `%autoreload` configurada na primeira célula, qualquer alteração que você fizer nos arquivos dentro de `src/` será atualizada automaticamente no ambiente do Notebook.

## 📊 Fluxo do Pipeline & Métricas de Avaliação

### 1. Preparação (Engenharia de Features)

- Agregação realizada rigorosamente sob a chave `customer_unique_id`.
- Extração de métricas de **RFM** (Recência, Frequência, Valor Monetário) combinadas com características operacionais (Nota de avaliação média, valor total do frete e parcelas médias).
- Aplicação de `np.log1p` para mitigar o efeito da cauda longa e distribuição assimétrica do e-commerce.

### 2. Redução de Espaço (PCA)

- Avaliação por **Variância Explicada Acumulada** para determinar o número ideal de componentes principais que retêm a informação essencial (foco entre 80% e 95% de variância).

### 3. Agrupamento Rígido (K-Means)

- **Método do Cotovelo (Inércia/WCSS):** Identificação visual da quebra de ganho de variância interna.
- **Coeficiente de Silhueta:** Avaliação da coesão interna e separabilidade dos grupos formados.

### 4. Agrupamento Probabilístico (GMM)

- **BIC (Bayesian Information Criterion) & AIC (Akaike Information Criterion):** Critérios de informação usados para selecionar o número de componentes, penalizando a complexidade para mitigar _overfitting_.

### 5. Comparação e Validação de Negócio

- **ARI (Adjusted Rand Index) & NMI (Normalized Mutual Information):** Métricas para avaliar a concordância matemática das fronteiras geradas pelos dois algoritmos.
- **Análise de Perfilamento:** Tradução das coordenadas dos clusters de volta para a escala monetária/temporal real para a criação de personas de e-commerce (ex: Clientes VIP, Churn em Potencial, Clientes Recentes de Baixo Ticket).
