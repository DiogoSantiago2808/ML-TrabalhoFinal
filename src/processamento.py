import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def carregar_e_agregar_dados(caminho_pasta: str) -> pd.DataFrame:
    """
    Passo 1: Carrega os CSVs da Olist, faz os merges necessários
    e agrega tudo por 'customer_unique_id'.
    """
    # 1. Carregar os datasets apontando para a pasta 'dados/'
    # O pandas adiciona automaticamente o .csv se você preferir, mas vamos colocar a string completa:
    customers = pd.read_csv(f'{caminho_pasta}olist_customers_dataset.csv')
    orders = pd.read_csv(f'{caminho_pasta}olist_orders_dataset.csv')
    payments = pd.read_csv(f'{caminho_pasta}olist_order_payments_dataset.csv')
    reviews = pd.read_csv(f'{caminho_pasta}olist_order_reviews_dataset.csv')
    items = pd.read_csv(f'{caminho_pasta}olist_order_items_dataset.csv')

    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    # 2. Pré-agregação por pedido
    payments_agg = payments.groupby('order_id').agg({'payment_value': 'sum', 'payment_installments': 'max'}).reset_index()
    reviews_agg = reviews.groupby('order_id').agg({'review_score': 'mean'}).reset_index()
    items_agg = items.groupby('order_id').agg({'freight_value': 'sum'}).reset_index()

    # 3. Cruzamento (Merges)
    df_orders = pd.merge(orders, customers, on='customer_id', how='inner')
    df_orders = df_orders[~df_orders['order_status'].isin(['canceled', 'unavailable'])]
    
    df_orders = pd.merge(df_orders, payments_agg, on='order_id', how='left')
    df_orders = pd.merge(df_orders, reviews_agg, on='order_id', how='left')
    df_orders = pd.merge(df_orders, items_agg, on='order_id', how='left')

    # Preenchimento de nulos
    df_orders['payment_value'] = df_orders['payment_value'].fillna(0)
    df_orders['freight_value'] = df_orders['freight_value'].fillna(0)
    df_orders['payment_installments'] = df_orders['payment_installments'].fillna(1)

    # 4. Cálculo de Recência
    snapshot_date = df_orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    df_orders['days_since_purchase'] = (snapshot_date - df_orders['order_purchase_timestamp']).dt.days

    # 5. Agregação por cliente único
    customer_df = df_orders.groupby('customer_unique_id').agg({
        'days_since_purchase': 'min',
        'order_id': 'nunique',
        'payment_value': 'sum',
        'payment_installments': 'mean',
        'freight_value': 'sum',
        'review_score': 'mean'
    }).reset_index()

    customer_df.rename(columns={
        'days_since_purchase': 'recencia',
        'order_id': 'frequencia',
        'payment_value': 'monetario',
        'payment_installments': 'parcelas_medias',
        'freight_value': 'frete_total',
        'review_score': 'review_medio'
    }, inplace=True)

    customer_df['review_medio'] = customer_df['review_medio'].fillna(customer_df['review_medio'].mean())

    return customer_df


def tratar_e_padronizar(df: pd.DataFrame, colunas_features: list) -> np.ndarray:
    # Deixamos essa aqui vazia por enquanto ou você já quer implementar junto?
    pass