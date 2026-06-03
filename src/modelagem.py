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
    # 1. Criar um loop de 2 até k_max colhendo a inércia e silhouette_score
    # 2. Plotar o gráfico do Cotovelo (Inércia vs K)
    # 3. Plotar o gráfico da Silhueta (Score vs K)
    # 4. Instanciar e ajustar o KMeans final usando o k_escolhido
    
    kmeans_modelo = None # Substitua pelo modelo ajustado
    labels = np.array([]) # Substitua pelos labels finais (.labels_)
    return kmeans_modelo, labels


def rodar_gmm(dados_pca: np.ndarray, n_max: int = 10, n_escolhido: int = 4):
    """
    Passo 5: Roda o GMM de 2 até n_max, calcula/plota as métricas
    de BIC e AIC, e retorna o modelo final ajustado com o n_escolhido.
    """
    # 1. Criar um loop de 2 até n_max colhendo o .bic() e .aic() do modelo
    # 2. Plotar o gráfico de linhas comparando as curvas do BIC e AIC
    # 3. Instanciar e ajustar o GaussianMixture final usando o n_escolhido
    
    gmm_modelo = None # Substitua pelo modelo ajustado
    labels = np.array([]) # Substitua pelos labels finais (.predict)
    return gmm_modelo, labels


def comparar_modelos(labels_kmeans: np.ndarray, labels_gmm: np.ndarray) -> None:
    """
    Passo 6: Avalia o nível de concordância estatística entre as duas partições.
    """
    # 1. Calcular o ARI (adjusted_rand_score)
    # 2. Calcular o NMI (normalized_mutual_info_score)
    # 3. Printar os resultados na tela
    # 4. Exibir a matriz de contingência cruzada usando pd.crosstab
    
    pass


def analisar_perfis(df_original: pd.DataFrame, agrupado_por: str) -> pd.DataFrame:
    """
    Passo 7: Agrupa a base de clientes original pelas labels geradas e calcula
    as médias/medianas reais para interpretação de negócios.
    """
    # 1. Usar o df_original.groupby(agrupado_por)
    # 2. Calcular .mean() ou .median() das variáveis de negócio (gastos, dias, notas)
    
    resumo_perfis = pd.DataFrame() # Substitua pelo agrupamento final
    return resumo_perfis