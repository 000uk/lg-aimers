import pandas as pd
from preprocessing import (
  load_data, add_date_features, add_holiday_info,
  fit_label_encoders, save_encoders, encode_labels,
)
from stl import stl_decompose, extrapolate_trend, repeat_seasonal
from dataset import Custom_Dataset, time_based_split

DATA_PATH = "data/train/train.csv"

# -----------------------------
# 1) 데이터 전처리
# -----------------------------
df_train, stats = load_data(pd.read_csv(DATA_PATH))

df_train = add_date_features(df_train)
df_train = add_holiday_info(df_train)

le_store, le_menu, le_holiday = fit_label_encoders(df_train)
save_encoders(le_store, le_menu, le_holiday, 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl')
df_train = encode_labels(df_train, le_store, le_menu, le_holiday)

df_train = df_train.drop(columns=['store', 'menu', 'day', 'month', 'dow', 'week','day_of_year', 'holiday']).fillna(0)
df_train = stl_decompose(df_train)

# -----------------------------
# 2) 데이터셋 생성(윈도우 등) 및 분리
#    - 역할: 데이터를 인덱스로 접근 가능하게 래핑(wrapping)
#    - 슬라이딩 윈도우, 전처리, feature 선택 등 모든 샘플 단위 전처리를 여기서 수행
# -----------------------------
dataset = Custom_Dataset(
    df=df_train,
    input_len=28,
    pred_len=7,
    id_cols=("store_enc", "menu_enc"),
    target_col="residual"
)

# 전체 날짜 기준으로 split임
# windows = {
#     "X_enc": dataset.X_enc,
#     "X_dec_future": dataset.X_dec_future,
#     "y_resid": dataset.y_resid,
#     "trend_future": dataset.trend_future,
#     "seasonal_future": dataset.seasonal_future,
#     "info": None  # info가 필요 없으면 None
# }
# meta = dataset.meta
# split = time_based_split(windows, meta, val_ratio=0.1)


# 임베딩은 맨 마지막에 할까? 모델 넣을 때 달라지는 거니까
# 모델 학습 코드 일단은 patchTFT로만 해보자
model = Transformer(...)  
model.fit(X_train, y_train)

# -----------------------------
# ?) 최종 예측
# -----------------------------
future_residual = model.predict(future_input)
future_trend = extrapolate_trend(df_train['trend'], horizon=7) # - Trend: 선형/다항 회귀로 앞으로 연장
future_seasonal = repeat_seasonal(df_train['seasonal'], horizon=7) # - Seasonal: 주기 반복
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