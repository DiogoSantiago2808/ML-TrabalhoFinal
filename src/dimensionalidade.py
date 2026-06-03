import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def aplicar_pca(dados_padronizados: np.ndarray) -> PCA:
    """
    Passo 3 (Parte A): Instancia e ajusta o modelo PCA nos dados padronizados
    para que você possa analisar a variância.
    """
    # 1. Instanciar o PCA (sem definir n_components para avaliar todas as dimensões)
    # 2. Ajustar (.fit) nos dados_padronizados
    
    # Substitua pelo objeto do PCA já ajustado
    pca_modelo = PCA() 
    return pca_modelo


def plotar_variancia_pca(pca_modelo: PCA) -> None:
    """
    Passo 3 (Parte B): Gera o gráfico de Linha/Barras mostrando a
    variância explicada acumulada para ajudar na escolha dos componentes.
    """
    # 1. Calcular a variância acumulada usando np.cumsum(pca_modelo.explained_variance_ratio_)
    # 2. Criar um gráfico usando matplotlib/seaborn
    # 3. Adicionar títulos, labels nos eixos e uma linha guia (ex: em 90% da variância)
    # 4. Dar plt.show()
    
    pass # Função apenas visual, não precisa retornar nada