import torch
import torch.nn as nn



# store_menu, store, menu 개수 확인
num_store_menu = df['store_menu_enc'].nunique()
num_store = df['store_enc'].nunique()
num_menu = df['menu_enc'].nunique()

# 임베딩 차원 설정 (보통 min(50, N/2) 정도)
emb_dim_store_menu = min(50, num_store_menu // 2)
emb_dim_store = min(20, num_store // 2)
emb_dim_menu = min(20, num_menu // 2)

# 임베딩 레이어 정의
embedding_store_menu = nn.Embedding(num_embeddings=num_store_menu, embedding_dim=emb_dim_store_menu)
embedding_store = nn.Embedding(num_embeddings=num_store, embedding_dim=emb_dim_store)
embedding_menu = nn.Embedding(num_embeddings=num_menu, embedding_dim=emb_dim_menu)

# 예시: 데이터 배치
store_menu_idx = torch.tensor(df['store_menu_enc'].values)
store_idx = torch.tensor(df['store_enc'].values)
menu_idx = torch.tensor(df['menu_enc'].values)

# 임베딩 적용
store_menu_emb = embedding_store_menu(store_menu_idx)
store_emb = embedding_store(store_idx)
menu_emb = embedding_menu(menu_idx)

# 모델 입력으로 사용 가능 (concat 등)
x_cat = torch.cat([store_menu_emb, store_emb, menu_emb], dim=1)