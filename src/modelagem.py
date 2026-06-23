import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def rodar_kmeans(dados_famd: np.ndarray, k_max: int = 10, k_escolhido: int = 4):
    inercias = []
    silhuetas = []
    ks = range(2, k_max + 1)
    
    print("[K-Means] A avaliar métricas de hiperparâmetros...")
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_temp = kmeans.fit_predict(dados_famd)
        
        inercias.append(kmeans.inertia_)
        
        score_sil = silhouette_score(dados_famd, labels_temp, sample_size=10000, random_state=42)
        silhuetas.append(score_sil)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(ks, inercias, marker='o', linestyle='--', color='b', linewidth=2)
    ax1.set_title('Método do Cotovelo (Inércia)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia Total')
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(ks, silhuetas, marker='o', linestyle='--', color='orange', linewidth=2)
    ax2.set_title('Coeficiente de Silhueta Médio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Score de Silhueta')
    ax2.set_xticks(ks)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    print(f"[K-Means] A ajustar o modelo definitivo com K = {k_escolhido}...")
    kmeans_modelo = KMeans(n_clusters=k_escolhido, n_init=10, random_state=42)
    labels = kmeans_modelo.fit_predict(dados_famd)
    
    return kmeans_modelo, labels


def rodar_gmm(dados_famd: np.ndarray, n_max: int = 10, n_escolhido: int = 4):
    bics = []
    aics = []
    ns = range(2, n_max + 1)
    
    print("[GMM] A avaliar curvas de densidade probabilística...")
    for n in ns:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(dados_famd)
        
        bics.append(gmm.bic(dados_famd))
        aics.append(gmm.aic(dados_famd))
        
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

    print(f"[GMM] A ajustar o modelo definitivo com N = {n_escolhido}...")
    gmm_modelo = GaussianMixture(n_components=n_escolhido, random_state=42)
    gmm_modelo.fit(dados_famd)
    labels = gmm_modelo.predict(dados_famd)
    
    return gmm_modelo, labels


def comparar_modelos(labels_kmeans: np.ndarray, labels_gmm: np.ndarray) -> None:
    ari = adjusted_rand_score(labels_kmeans, labels_gmm)
    nmi = normalized_mutual_info_score(labels_kmeans, labels_gmm)
    
    print("\n" + "="*20 + " COMPARAÇÃO ESTATÍSTICA DE CONCORDÂNCIA " + "="*20)
    print(f"• Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"• Normalized Mutual Information (NMI): {nmi:.4f}")
    print("-"*81)
    print("Significado: Valores próximos de 1.0 indicam que os modelos agruparam os mesmos clientes.")
    print("Valores próximos de 0.0 indicam que as partições foram feitas de forma independente.")
    print("="*81 + "\n")
    
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
    resumo_perfis = df_original.groupby(agrupado_por).mean(numeric_only=True)
    
    contagem_clientes = df_original[agrupado_por].value_counts()
    percentual_clientes = df_original[agrupado_por].value_counts(normalize=True) * 100
    
    resumo_perfis.insert(0, 'qtd_clientes', contagem_clientes)
    resumo_perfis.insert(1, 'pct_clientes', percentual_clientes)
    
    resumo_perfis.sort_index(inplace=True)
    
    return resumo_perfis