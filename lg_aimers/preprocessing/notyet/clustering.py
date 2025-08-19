import torch
import torch.nn as nn

# Assuming df_dropped is available from previous steps and contains 'store_enc' and 'menu_enc'
# Create tensors from the encoded columns
store_idx = torch.tensor(df_dropped['store_enc'].values)
menu_idx = torch.tensor(df_dropped['menu_enc'].values)

# 2. Embedding Layer (사전 학습용)
# Use the number of unique values in the encoded columns for embedding size
store_emb_layer = nn.Embedding(len(df_dropped['store_enc'].unique()), 4)
menu_emb_layer  = nn.Embedding(len(df_dropped['menu_enc'].unique()), 4)


# 3. 임베딩 추출 (사전 학습된 matrix처럼 사용)
store_emb_matrix = store_emb_layer(store_idx)
menu_emb_matrix  = menu_emb_layer(menu_idx)

store_emb_agg = df_dropped.groupby('store_enc').apply(
    lambda x: menu_emb_matrix[x.index].detach().mean(dim=0)
)

from sklearn.cluster import KMeans
import numpy as np

X = np.stack(store_emb_agg.values)  # (num_stores, emb_dim)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
store_cluster_labels = kmeans.labels_