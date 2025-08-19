import pandas as pd
from preprocessing import (
  load_data, add_date_features, add_holiday_info,
  fit_label_encoders, save_encoders, encode_labels,
  sales_scaler
)

DATA_PATH = "data/train/train.csv"

df_train = pd.read_csv(DATA_PATH)
df_train = load_data(df_train)

"""
1. 데이터셋 전처리
"""
df_train = add_date_features(df_train)
df_train = add_holiday_info(df_train)

le_store, le_menu, le_holiday = fit_label_encoders(df_train)
save_encoders(le_store, le_menu, le_holiday, 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl')
df_train = encode_labels(df_train, le_store, le_menu, le_holiday)

df_train = df_train.fillna(0)
df_dropped = df_train.drop(columns=['store', 'menu', 'day', 'month', 'dow', 'week','day_of_year', 'holiday'])
"""
2. 
"""
# df_train = stl_decompose(df_train)
# df_train = add_outlier_flag(df_train, window=28)

# 임베딩은 맨 마지막에 할까? 모델 넣을 때 달라지는 거니까

# # 학습에 필요없는 거 drop
# df_train = df_train.drop(columns=['date'])
# df_train = df_train.drop(columns=['store_menu'])

# 모델 학습 코드 일단은 patchTFT로만 해보자
