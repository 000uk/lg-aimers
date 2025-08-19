import torch
import torch.nn as nn

class CategoryEmbeddings(nn.Module):
    def __init__(self, df, le_store, le_menu, le_holiday):
        super().__init__()
        # store_menu, store, menu 개수 확인
        self.num_store = df['store_enc'].nunique()
        self.num_menu = df['menu_enc'].nunique()
        self.num_holiday = df['holiday_enc'].nunique()
        
        # 임베딩 차원 설정 (보통 min(50, N//2) 정도)
        self.emb_dim_store = min(20, self.num_store // 2)
        self.emb_dim_menu = min(20, self.num_menu // 2)
        self.emb_dim_holiday = min(50, self.num_holiday // 2)
        
        # 임베딩 레이어 정의
        self.embedding_store = nn.Embedding(self.num_store, self.emb_dim_store)
        self.embedding_menu = nn.Embedding(self.num_menu, self.emb_dim_menu)
        self.embedding_holiday = nn.Embedding(self.num_holiday, self.emb_dim_holiday)
        
        # label encoders (UNK-safe 인덱스 처리 가능)
        self.le_store = le_store
        self.le_menu = le_menu
        self.le_holiday = le_holiday
    
    def forward(self, df_batch):
        # UNK-safe 변환 함수
        def safe_transform(le, series):
            vals = []
            for v in series:
                if v in le.classes_:
                    vals.append(le.transform([v])[0])
                else:
                    vals.append(len(le.classes_))  # UNK index
            return torch.tensor(vals, dtype=torch.long)
        
        store_idx = safe_transform(self.le_store, df_batch['store'])
        menu_idx = safe_transform(self.le_menu, df_batch['menu'])
        holiday_idx = safe_transform(self.le_holiday, df_batch['holiday'])
        
        # 임베딩 적용
        holiday_emb = self.embedding_holiday(holiday_idx)
        store_emb = self.embedding_store(store_idx)
        menu_emb = self.embedding_menu(menu_idx)
        
        # concat 후 모델 입력
        x_cat = torch.cat([store_emb, menu_emb, holiday_emb], dim=1)
        return x_cat