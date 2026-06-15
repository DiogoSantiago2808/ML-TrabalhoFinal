import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def carregar_e_agregar_dados(caminho_pasta: str) -> pd.DataFrame:
    """
    Passo 1: Carga e Engenharia de Atributos Suprema Otimizada para FAMD.
    Transforma categorias esparsas em volume financeiro contínuo.
    """
    customers = pd.read_csv(f'{caminho_pasta}olist_customers_dataset.csv')
    orders = pd.read_csv(f'{caminho_pasta}olist_orders_dataset.csv')
    payments = pd.read_csv(f'{caminho_pasta}olist_order_payments_dataset.csv')
    reviews = pd.read_csv(f'{caminho_pasta}olist_order_reviews_dataset.csv')
    items = pd.read_csv(f'{caminho_pasta}olist_order_items_dataset.csv')
    products = pd.read_csv(f'{caminho_pasta}olist_products_dataset.csv')
    sellers = pd.read_csv(f'{caminho_pasta}olist_sellers_dataset.csv')

    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

    df_pedidos_base = pd.merge(orders, customers, on='customer_id', how='inner')
    df_pedidos_base = df_pedidos_base[~df_pedidos_base['order_status'].isin(['canceled', 'unavailable'])]

    # --- DETALHAMENTO DE PRODUTOS ---
    items_completo = pd.merge(items, products, on='product_id', how='left')
    items_completo = pd.merge(items_completo, sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    items_completo = pd.merge(items_completo, df_pedidos_base[['order_id', 'customer_state']], on='order_id', how='left')
    
    items_completo['is_vendedor_mesmo_estado'] = (items_completo['seller_state'] == items_completo['customer_state']).astype(int)

    # Mapeamento de Macro-Departamentos
    macro_categorias = {
        'gasto_casa_decoracao': ['cama_mesa_banho', 'utilidades_domesticas', 'moveis_decoracao', 'casa_conforto', 'moveis_escritorio'],
        'gasto_saude_beleza': ['beleza_saude', 'perfumaria', 'fraldas_higiene'],
        'gasto_esporte_lazer': ['esporte_lazer', 'brinquedos', 'games', 'instrumentos_musicais'],
        'gasto_tecnologia': ['informatica_acessorios', 'telefonia', 'eletronicos', 'sinalizacao_e_seguranca'],
        'gasto_moda_acessorios': ['relogios_presentes', 'fashion_bolsas_e_acessorios', 'malas_acessorios', 'calcados']
    }
    
    # REGRA 1: Multiplicamos a flag pelo valor do item para saber o valor gasto por categoria!
    for macro, sub_cats in macro_categorias.items():
        items_completo[macro] = items_completo['product_category_name'].isin(sub_cats).astype(int) * items_completo['price']

    dict_items_agg = {
        'freight_value': 'sum',
        'product_id': 'count', 
        'product_weight_g': 'mean',
        'product_photos_qty': 'mean',
        'seller_id': 'nunique',
        'is_vendedor_mesmo_estado': 'mean',
        **{macro: 'sum' for macro in macro_categorias.keys()}
    }
    items_agg = items_completo.groupby('order_id').agg(dict_items_agg).reset_index()
    items_agg.rename(columns={'product_id': 'qty_itens', 'seller_id': 'qtd_vendedores_pedido'}, inplace=True)

    # --- DETALHAMENTO DE PAGAMENTOS ---
    payments['is_boleto'] = (payments['payment_type'] == 'boleto').astype(int)
    payments['is_cartao_credito'] = (payments['payment_type'] == 'credit_card').astype(int)
    
    payments_agg = payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'max',
        'is_boleto': 'max',
        'is_cartao_credito': 'max'
    }).reset_index()

    # --- AVALIAÇÕES ---
    reviews['has_comment_text'] = reviews['review_comment_message'].notna().astype(int)
    reviews_agg = reviews.groupby('order_id').agg({
        'review_score': 'mean',
        'has_comment_text': 'max'
    }).reset_index()

    # --- UNIÃO MÃE ---
    df_orders = pd.merge(df_pedidos_base, payments_agg, on='order_id', how='left')
    df_orders = pd.merge(df_orders, reviews_agg, on='order_id', how='left')
    df_orders = pd.merge(df_orders, items_agg, on='order_id', how='left')

    df_orders['tempo_entrega'] = (df_orders['order_delivered_customer_date'] - df_orders['order_purchase_timestamp']).dt.days
    df_orders['is_weekend'] = df_orders['order_purchase_timestamp'].dt.weekday.isin([5, 6]).astype(int)
    df_orders['is_sudeste'] = df_orders['customer_state'].isin(['SP', 'RJ', 'MG', 'ES']).astype(int)

    # Preenchimento de Nulos
    df_orders['payment_value'] = df_orders['payment_value'].fillna(0)
    df_orders['freight_value'] = df_orders['freight_value'].fillna(0)
    df_orders['payment_installments'] = df_orders['payment_installments'].fillna(1)
    df_orders['tempo_entrega'] = df_orders['tempo_entrega'].fillna(df_orders['tempo_entrega'].median())
    df_orders['product_weight_g'] = df_orders['product_weight_g'].fillna(df_orders['product_weight_g'].median())
    df_orders['qty_itens'] = df_orders['qty_itens'].fillna(1)
    
    colunas_zero = ['is_boleto', 'is_cartao_credito', 'has_comment_text'] + list(macro_categorias.keys())
    for col in colunas_zero:
        df_orders[col] = df_orders[col].fillna(0)

    snapshot_date = df_orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    df_orders['days_since_purchase'] = (snapshot_date - df_orders['order_purchase_timestamp']).dt.days

    # --- AGREGAÇÃO FINAL POR CLIENTE ---
    dict_customer_agg = {
        'days_since_purchase': 'min',
        'order_id': 'nunique',
        'payment_value': 'sum',
        'payment_installments': 'mean',
        'is_boleto': 'mean',
        'is_cartao_credito': 'mean',
        'freight_value': 'sum',
        'review_score': 'mean',
        'has_comment_text': 'max', # Mantido como flag pura (0 ou 1)
        'tempo_entrega': 'mean',
        'is_weekend': 'mean',
        'is_sudeste': 'max',       # Mantido como flag pura (0 ou 1)
        'product_weight_g': 'mean',
        'qty_itens': 'sum',
        **{macro: 'sum' for macro in macro_categorias.keys()}
    }
    
    customer_df = df_orders.groupby('customer_unique_id').agg(dict_customer_agg).reset_index()

    novos_nomes = {
        'days_since_purchase': 'recencia',
        'order_id': 'frequencia',
        'payment_value': 'monetario',
        'payment_installments': 'parcelas_medias',
        'is_boleto': 'pct_boleto',
        'is_cartao_credito': 'pct_cartao_credito',
        'freight_value': 'frete_total',
        'review_score': 'review_medio',
        'has_comment_text': 'escreveu_comentario',
        'tempo_entrega': 'tempo_entrega_medio',
        'is_weekend': 'pct_compras_fds',
        'product_weight_g': 'peso_medio_produtos',
        'qty_itens': 'qtd_itens_total'
    }
    customer_df.rename(columns=novos_nomes, inplace=True)
    customer_df['review_medio'] = customer_df['review_medio'].fillna(customer_df['review_medio'].mean())

    return customer_df

def tratar_e_padronizar(df: pd.DataFrame, colunas_features: list) -> np.ndarray:
    """
    Passo 2: Apenas aplica log para corrigir caudas longas de colunas financeiras e volumétricas.
    """
    df_copia = df[colunas_features].copy()
    colunas_log = ['recencia', 'frequencia', 'monetario', 'frete_total', 'tempo_entrega_medio', 'peso_medio_produtos', 'qtd_itens_total']
    
    for col in colunas_log:
        if col in df_copia.columns:
            df_copia[col] = np.log1p(np.clip(df_copia[col], 0, None))
            
    return df_copia # Retorna o DataFrame tratado (o FAMD fará o scaler internamente)