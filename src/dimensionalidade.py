import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prince

def aplicar_famd(df: pd.DataFrame, colunas_features: list, n_componentes: int = 12) -> tuple:
    df_famd = df[colunas_features].copy()
    
    colunas_categoricas = ['escreveu_comentario', 'is_sudeste']
    for col in colunas_categoricas:
        if col in df_famd.columns:
            df_famd[col] = df_famd[col].astype(str)
            
    famd_modelo = prince.FAMD(
        n_components=n_componentes,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    famd_modelo.fit(df_famd)
    dados_transformados = famd_modelo.row_coordinates(df_famd).to_numpy()
    
    print(f"[FAMD] Dados reduzidos com sucesso para {n_componentes} eixos.")
    return famd_modelo, dados_transformados


def encontrar_melhor_k_famd(famd_modelo) -> int:
    eigenvalues = np.array(famd_modelo.eigenvalues_)
    k_kaiser = np.sum(eigenvalues > 1.0)
    
    variancias = np.array(famd_modelo.percentage_of_variance_) / 100
    media_ganho = np.mean(variancias)
    
    k_cotovelo = 3
    for i in range(1, len(variancias)):
        if variancias[i] < (media_ganho * 0.90):
            k_cotovelo = i
            break
            
    if 2 <= k_kaiser <= 6:
        n_recomendado = int(k_kaiser)
        metodo = "Critério de Kaiser (Autovalores > 1.0)"
    else:
        n_recomendado = int(k_cotovelo)
        metodo = "Método do Cotovelo (Desaceleração do Ganho Marginal)"
        
    print(f"\n[💡 ANÁLISE AUTOMÁTICA] Método aplicado: {metodo}")
    print(f"👉 O número ideal de componentes para sua clusterização é: {n_recomendado} componentes.\n")
    
    return n_recomendado


def plotar_variancia_famd(famd_modelo, n_recomendado: int = None) -> None:
    variancia_individual = [v / 100 for v in famd_modelo.percentage_of_variance_]
    variancia_acumulada = np.cumsum(variancia_individual)
    
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(variancia_acumulada) + 1), 
        variancia_acumulada, 
        marker='o', 
        linestyle='--', 
        color='g',
        linewidth=2,
        label='Variância Acumulada'
    )
    
    if n_recomendado:
        plt.axvline(
            x=n_recomendado, 
            color='r', 
            linestyle=':', 
            linewidth=2, 
            label=f'Corte Sugerido ({n_recomendado} eixos)'
        )
        plt.scatter(
            n_recomendado, 
            variancia_acumulada[n_recomendado - 1], 
            color='red', 
            s=120, 
            zorder=5
        )
    
    plt.title('Variância Explicada Acumulada pelo FAMD (Dados Otimizados)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Componentes Principais (FAMD)', fontsize=12)
    plt.ylabel('Variância Explicada Acumulada', fontsize=12)
    plt.xticks(range(1, len(variancia_acumulada) + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.show()