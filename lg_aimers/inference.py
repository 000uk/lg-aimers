import glob, os
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

# -----------------------------
# 1) 저장된 인코더 불러오기
# -----------------------------
le_store_menu, le_store, le_menu, le_holiday = load_encoders(
    'le_store_menu.pkl', 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl'
)

# -----------------------------
# 2) Test 데이터 로드 & 전처리
# -----------------------------
# TEST_PATH = "data/test"
# test_files = sorted(glob.glob(os.path.join(TEST_PATH, "TEST_*.csv")))

# test_dfs = [pd.read_csv(f) for f in test_files]
# df_test = pd.concat(test_dfs, axis=0).reset_index(drop=True)

TEST_PATH = "data/test"
test_files = sorted(glob.glob(f"{TEST_PATH}/TEST_*.csv"))

df_tests = []
for f in test_files:
    df, stats = load_data(pd.read_csv(TEST_PATH))
    df["file_id"] = os.path.basename(f).replace(".csv", "")
    df_tests.append(df)
df_test = pd.concat(df_tests, ignore_index=True)

# (train과 동일한 feature 전처리)
df_test = add_date_features(df_test)
df_test = add_holiday_info(df_test)

# 인코딩
df_test = encode_labels(df_test, le_store_menu, le_store, le_menu, le_holiday)

# 불필요 컬럼 제거
df_test = df_test.drop(columns=[
    'store', 'menu', 'day', 'month', 'dow', 'week',
    'day_of_year', 'holiday'
]).fillna(0)

# STL 분해 (trend, seasonal, residual 분리)
df_test = stl_decompose(df_test)

# -----------------------------
# 3) Test Dataset 생성
# -----------------------------
test_dataset = WindowGenerator(
    df=df_test,
    input_len=28,
    pred_len=7,
    id_cols=("store_menu_enc", "menu_enc"),
    target_col=None  # target 없는 경우
)

test_windows = {
    "X_enc": test_dataset.X_enc,
    "X_dec_future": test_dataset.X_dec_future,
    "store_menu_id": test_dataset.meta["store_menu_enc"].values,
    "menu_id": test_dataset.meta["menu_enc"].values,
}

# -----------------------------
# 4) 모델 불러오기
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

num_store_menus = int(df_test["store_menu_enc"].max()) + 1
num_menus = int(df_test["menu_enc"].max()) + 1
emb_dim = 2

X_enc_features = test_dataset.X_enc.shape[-1]
X_dec_features = test_dataset.X_dec_future.shape[-1]

model = SimpleTransformer(
    X_enc_features, X_dec_features,
    num_store_menus=num_store_menus, num_menus=num_menus,
    emb_dim=emb_dim,
    d_model=64, nhead=4, num_layers=2, dropout=0.5
).to(device)

model.load_state_dict(torch.load("/content/best_model_step_28.pt", map_location=device))
model.eval()

# -----------------------------
# 5) 예측
# -----------------------------
preds = []
for batch in test_loader:
    with torch.no_grad():
        y_hat = model(
            batch["X_enc"].to(device),
            batch["X_dec_future"].to(device),
            batch["store_menu_id"].to(device),
            batch["menu_id"].to(device)
        )
        preds.append(y_hat.cpu().numpy())

preds = np.concatenate(preds, axis=0)  # [전체 샘플, 7일]
print("예측 결과 shape:", preds.shape)




















# 제출용: 원본 정보 + 예측 결과
submission = df_test[['date', 'store_menu']].copy()
submission['predicted_sales'] = preds
submission.to_csv('submission.csv', index=False)


# 추론용 df_test
df_submit = df_test[['store_menu']].copy()
df_submit['pred'] = model.predict(X_test)
df_submit.to_csv('submission.csv', index=False)