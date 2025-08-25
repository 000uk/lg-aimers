import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from preprocessing import (
  load_data, add_date_features, add_holiday_info,
  fit_label_encoders, save_encoders, load_encoders, encode_labels
)
from stl import stl_decompose, extrapolate_trend, repeat_seasonal
from dataset import WindowGenerator, TSFullDataset, time_based_split, rolling_split
from model import SimpleTransformer

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        X_enc = batch["X_enc"].to(device)
        X_dec = batch["X_dec_future"].to(device)
        y_full = batch["y_full"].to(device)
        store_menu_id = batch["store_menu_id"].to(device)
        menu_id = batch["menu_id"].to(device)

        optimizer.zero_grad()
        output = model(X_enc, X_dec, store_menu_id, menu_id, y_prev=y_full)
        loss = criterion(output, y_full)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    
    model.eval()
    val_loss_teacher = 0
    val_loss_rollout = 0
    with torch.no_grad():
        for batch in loader:
            X_enc = batch["X_enc"].to(device)
            X_dec = batch["X_dec_future"].to(device)
            y_full = batch["y_full"].to(device)
            store_menu_id = batch["store_menu_id"].to(device)
            menu_id = batch["menu_id"].to(device)

            # -------- Teacher Forcing --------
            output_teacher = model(X_enc, X_dec, store_menu_id, menu_id, y_prev=y_full)
            val_loss_teacher += criterion(output_teacher, y_full).item()

            # -------- Autoregressive Rollout --------
            B, dec_len, _ = X_dec.shape
            y_prev = torch.zeros(B, dec_len, device=device)
            preds = []
            for t in range(dec_len):
                out_t = model(
                    X_enc,
                    X_dec[:, :t+1, :],
                    store_menu_id,
                    menu_id,
                    y_prev=y_prev[:, :t+1]
                )
                preds_t = out_t[:, t]
                preds.append(preds_t)
                y_prev[:, t] = preds_t

            y_hat = torch.stack(preds, dim=1)
            val_loss_rollout += criterion(y_hat, y_full).item()

    return val_loss_teacher / len(loader), val_loss_rollout / len(loader)
DATA_PATH = "data/train/train.csv"

# -----------------------------
# 1) 데이터 전처리
# -----------------------------
df_train, stats = load_data(pd.read_csv(DATA_PATH))

df_train = add_date_features(df_train)
df_train = add_holiday_info(df_train)

le_store_menu, le_store, le_menu, le_holiday = fit_label_encoders(df_train)
save_encoders(le_store_menu, le_store, le_menu, le_holiday, 'le_store_menu.pkl', 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl')
df_train = encode_labels(df_train, le_store_menu, le_store, le_menu, le_holiday)

df_train = df_train.drop(columns=['store', 'menu', 'day', 'month', 'dow', 'week','day_of_year', 'holiday']).fillna(0)
df_train = stl_decompose(df_train)

# -----------------------------
# 2) 데이터셋 생성(윈도우 등) 및 분리
#    - 역할: 데이터를 인덱스로 접근 가능하게 래핑(wrapping)
#    - 슬라이딩 윈도우, 전처리, feature 선택 등 모든 샘플 단위 전처리를 여기서 수행
# -----------------------------
dataset = WindowGenerator(
    df=df_train,
    input_len=28,
    pred_len=7,
    id_cols=("store_menu_enc", "menu_enc"),
    target_col="residual"
)
windows = {
    "X_enc": dataset.X_enc,
    "X_dec_future": dataset.X_dec_future,
    "y_resid": dataset.y_resid,
    "trend_future": dataset.trend_future,
    "seasonal_future": dataset.seasonal_future,
    "store_menu_id": dataset.meta["store_menu_enc"].values,
    "menu_id": dataset.meta["menu_enc"].values,
    "info": None  # info가 필요 없으면 None
}
meta = dataset.meta

# 전체 날짜 기준으로 split임
splits = rolling_split(windows, meta, train_len=365, val_len=28, step=28)
#splits = rolling_split(windows, meta, train_len=365, val_len=28, step=7)

# -------------------------------
# 3) 모델 생성
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

num_store_menus = int(df_train["store_menu_enc"].max()) + 1
num_menus = int(df_train["menu_enc"].max()) + 1
emb_dim = 2 # 임베딩 차원이 너무 놓으면 과적합 될 수도 있음

X_enc_features = splits[0]["train_X_enc"].shape[-1]
X_dec_features = splits[0]["train_X_dec_future"].shape[-1]

model = SimpleTransformer(
    X_enc_features, X_dec_features,
    num_store_menus=num_store_menus, num_menus=num_menus,
    emb_dim=emb_dim,
    d_model=64, nhead=4, num_layers=2, dropout=0.5
).to(device)





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