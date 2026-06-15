import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def rodar_kmeans(dados_pca: np.ndarray, k_max: int = 10, k_escolhido: int = 4):
    """
    Passo 4: Roda o K-Means de 2 até k_max, calcula/plota as métricas
    de Inércia e Silhueta, e retorna o modelo final ajustado com o k_escolhido.
    """
    inercias = []
    silhuetas = []
    ks = range(2, k_max + 1)
    
    print("[K-Means] A avaliar métricas de hiperparâmetros...")
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_temp = kmeans.fit_predict(dados_pca)
        
        inercias.append(kmeans.inertia_)
        
        # Subamostragem de segurança para evitar estouro de memória RAM (MemoryError) com 95k linhas
        score_sil = silhouette_score(dados_pca, labels_temp, sample_size=10000, random_state=42)
        silhuetas.append(score_sil)
        
    # Construção dos gráficos em subplots paralelos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico do Cotovelo
    ax1.plot(ks, inercias, marker='o', linestyle='--', color='b', linewidth=2)
    ax1.set_title('Método do Cotovelo (Inércia)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia Total')
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico da Silhueta
    ax2.plot(ks, silhuetas, marker='o', linestyle='--', color='orange', linewidth=2)
    ax2.set_title('Coeficiente de Silhueta Médio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Score de Silhueta')
    ax2.set_xticks(ks)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Ajuste do modelo definitivo selecionado
    print(f"[K-Means] A ajustar o modelo definitivo com K = {k_escolhido}...")
    kmeans_modelo = KMeans(n_clusters=k_escolhido, n_init=10, random_state=42)
    labels = kmeans_modelo.fit_predict(dados_pca)
    
    return kmeans_modelo, labels


def rodar_gmm(dados_pca: np.ndarray, n_max: int = 10, n_escolhido: int = 4):
    """
    Passo 5: Roda o GMM de 2 até n_max, calcula/plota as métricas
    de BIC e AIC, e retorna o modelo final ajustado com o n_escolhido.
    """
    bics = []
    aics = []
    ns = range(2, n_max + 1)
    
    print("[GMM] A avaliar curvas de densidade probabilística...")
    for n in ns:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(dados_pca)
        
        bics.append(gmm.bic(dados_pca))
        aics.append(gmm.aic(dados_pca))
        
    # Renderização das curvas AIC vs BIC
    plt.figure(figsize=(9, 5))
    plt.plot(ns, bics, marker='o', linestyle='-', color='crimson', linewidth=2, label='BIC (Bayesian Information Criterion)')
    plt.plot(ns, aics, marker='s', linestyle='--', color='dodgerblue', linewidth=2, label='AIC (Akaike Information Criterion)')
    
    plt.title('Critérios de Informação GMM (BIC vs AIC)', fontsize=13, fontweight='bold')
    plt.xlabel('Número de Componentes Mistos (N)')
    plt.ylabel('Valor do Critério (Menor é Melhor)')
    plt.xticks(ns)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # Ajuste do modelo probabilístico definitivo selecionado
    print(f"[GMM] A ajustar o modelo definitivo com N = {n_escolhido}...")
    gmm_modelo = GaussianMixture(n_components=n_escolhido, random_state=42)
    gmm_modelo.fit(dados_pca)
    labels = gmm_modelo.predict(dados_pca)
    
    return gmm_modelo, labels


def comparar_modelos(labels_kmeans: np.ndarray, labels_gmm: np.ndarray) -> None:
    """
    Passo 6: Avalia o nível de concordância estatística entre as duas partições.
    """
    # Cálculo dos índices estatísticos de validação externa
    ari = adjusted_rand_score(labels_kmeans, labels_gmm)
    nmi = normalized_mutual_info_score(labels_kmeans, labels_gmm)
    
    print("\n" + "="*20 + " COMPARAÇÃO ESTATÍSTICA DE CONCORDÂNCIA " + "="*20)
    print(f"• Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"• Normalized Mutual Information (NMI): {nmi:.4f}")
    print("-"*81)
    print("Significado: Valores próximos de 1.0 indicam que os modelos agruparam os mesmos clientes.")
    print("Valores próximos de 0.0 indicam que as partições foram feitas de forma independente.")
    print("="*81 + "\n")
    
    # Criação da matriz de contingência cruzada para análise visual do cruzamento de grupos
    matriz_contingencia = pd.crosstab(
        labels_kmeans, 
        labels_gmm, 
        rownames=['Clusters K-Means'], 
        colnames=['Clusters GMM']
    )
    
    print("Matriz de Contingência Cruzada (Distribuição de Clientes):")
    print(matriz_contingencia)
    print("\n")


def analisar_perfis(df_original: pd.DataFrame, agrupado_por: str) -> pd.DataFrame:
    """
    Passo 7: Agrupa a base de clientes original pelas labels geradas e calcula
    as médias reais para interpretação de negócios.
    """
    # Agrupamos pelo rótulo do cluster e extraímos as médias apenas de colunas numéricas reais
    # Note que isto ignora os logs aplicados no passo 2, trazendo métricas tangíveis ao negócio
    resumo_perfis = df_original.groupby(agrupado_por).mean(numeric_only=True)
    
    # Adicionamos uma coluna mostrando o volume absoluto e percentual de clientes por grupo
    contagem_clientes = df_original[agrupado_por].value_counts()
    percentual_clientes = df_original[agrupado_por].value_counts(normalize=True) * 100
    
    resumo_perfis.insert(0, 'qtd_clientes', contagem_clientes)
    resumo_perfis.insert(1, 'pct_clientes', percentual_clientes)
    
    # Ordena o DataFrame pelos índices dos clusters para facilitar a leitura paralela
    resumo_perfis.sort_index(inplace=True)
    
    return resumo_perfis