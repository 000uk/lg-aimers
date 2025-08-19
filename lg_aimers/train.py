import pandas as pd
from preprocessing import (
  load_data, add_date_features, add_holiday_info,
  fit_label_encoders, save_encoders, encode_labels,
)
from stl import stl_decompose, extrapolate_trend, repeat_seasonal

DATA_PATH = "data/train/train.csv"

# -----------------------------
# 1) 데이터셋 전처리
# -----------------------------
df_train, stats = load_data(pd.read_csv(DATA_PATH))

df_train = add_date_features(df_train)
df_train = add_holiday_info(df_train)

le_store, le_menu, le_holiday = fit_label_encoders(df_train)
save_encoders(le_store, le_menu, le_holiday, 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl')
df_train = encode_labels(df_train, le_store, le_menu, le_holiday)

df_train = df_train.drop(columns=['store', 'menu', 'day', 'month', 'dow', 'week','day_of_year', 'holiday']).fillna(0)

# -----------------------------
# 2) stl 분리 및 데이터셋 생성(윈도우 등)
# -----------------------------
df_train = stl_decompose(df_train)


x, y = Custom_Dataset(df['residual'])  # 슬라이딩 윈도우 시퀀스 만들기
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# 데이터셋 분리 time_based_split (학습 및 검증)

# 임베딩은 맨 마지막에 할까? 모델 넣을 때 달라지는 거니까
# 모델 학습 코드 일단은 patchTFT로만 해보자
model = Transformer(...)  
model.fit(X_train, y_train)

# -----------------------------
# ?) 최종 예측
# -----------------------------
future_residual = model.predict(future_input)
future_trend = extrapolate_trend(df['trend'], horizon) # - Trend: 선형/다항 회귀로 앞으로 연장
future_seasonal = repeat_seasonal(df['seasonal'], horizon) # - Seasonal: 주기 반복
future_pred = future_trend + future_seasonal + future_residual

# 원래 스키마로 되돌리기
output = pd.DataFrame({
    'date': future_dates,
    'store_menu': store_menu_id,  # 원래 인코딩/디코딩 필요
    'sales_qty_pred': future_pred
})

# 언젠가 정규화한거 복구해아하니까
# df = df.merge(stats, on='store_menu', how='left')
# df['sales_qty'] = df['sales_norm'] * df['std'] + df['mean']