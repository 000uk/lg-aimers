import pandas as pd
from preprocessing import load_data
from feature_engineering import stl_decompose

DATA_PATH = "data/train/train.csv"

df_train, le_store, le_menu = load_data(DATA_PATH)

df = stl_decompose(df_train)
df = add_eventy_flag(df, window=28)
df = add_outlier_flag(df, window=28)

# 모델 학습 코드
"""
주의: 원핫 인코딩과 임베딩용 라벨 인코딩을 동시에 쓰면, 
모델 종류에 따라 둘 중 하나만 쓰거나 분리해서 사용해야 함.
- 딥러닝 → 임베딩 사용
- 전통적 ML → 원핫 인코딩 사용
"""

# 학습이 끝나면 encoder 저장
import pickle
with open('le_store.pkl', 'wb') as f:
    pickle.dump(le_store, f)
with open('le_menu.pkl', 'wb') as f:
    pickle.dump(le_menu, f)