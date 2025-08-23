import numpy as np
import pandas as pd
from typing import Dict

def time_based_split(
    # custom_dataset에서 뽑아낸 학습용 배열 모음
    windows: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    val_ratio: float = 0.1
):
    # meta['pred_start_date'] 기준으로 시간순 정렬 후 뒤쪽 val_ratio를 검증으로 사용
    order = np.argsort(meta['pred_start_date'].values)
    for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future",
              "store_id", "menu_id", "store_menu_id"]:   # ← ID 추가
        windows[k] = windows[k][order]
    meta = meta.iloc[order].reset_index(drop=True)

    n = len(meta)
    n_val = max(1, int(np.round(n * val_ratio)))
    train_slice = slice(0, n - n_val)
    val_slice = slice(n - n_val, n)

    split = {}
    for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future",
              "store_id", "menu_id", "store_menu_id"]:  # ← ID 추가
        split[f"train_{k}"] = windows[k][train_slice]
        split[f"val_{k}"]   = windows[k][val_slice]

    split["train_meta"] = meta.iloc[0:n - n_val].reset_index(drop=True)
    split["val_meta"]   = meta.iloc[n - n_val:].reset_index(drop=True)
    split["info"] = windows.get("info", None)

    split["train_y_full"] = split["train_y_resid"] + split["train_trend_future"] + split["train_seasonal_future"]
    split["val_y_full"]   = split["val_y_resid"] + split["val_trend_future"] + split["val_seasonal_future"]

    return split

# 참고로 train_len은 input_len이랑은 다른 개념
# input_len은 지금은 28인데... 그걸 윈도우 별로 다 나눠놓은게 windowGenerator! loss 검증도 이거 기준
# train_len은 그런 window를 몇개를 써서 학습을 하겠댜는 말임
def rolling_split(windows, meta, train_len, val_len, step=7):
    """
    windows: dict of arrays (X_enc, X_dec_future, y_resid, trend_future, seasonal_future, store/menu IDs ...)
    meta: dataframe with pred_start_date
    train_len: 학습 기간 (윈도우 수)
    val_len: 검증 기간 (윈도우 수)
    step: 다음 윈도우로 이동할 간격 (pred_len과 맞추는 게 일반적)
    """
    order = np.argsort(meta['pred_start_date'].values)
    for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future",
              "store_id", "menu_id", "store_menu_id"]:  # ← ID 포함
        windows[k] = windows[k][order]
    meta = meta.iloc[order].reset_index(drop=True)

    splits = []
    n = len(meta)

    start = 0
    while start + train_len + val_len <= n:
        train_slice = slice(start, start + train_len)
        val_slice = slice(start + train_len, start + train_len + val_len)

        split = {}
        for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future",
                  "store_id", "menu_id", "store_menu_id"]:  # ← 동일하게 반영
            split[f"train_{k}"] = windows[k][train_slice]
            split[f"val_{k}"]   = windows[k][val_slice]

        split["train_meta"] = meta.iloc[train_slice].reset_index(drop=True)
        split["val_meta"]   = meta.iloc[val_slice].reset_index(drop=True)

        split["train_y_full"] = (
            split["train_y_resid"] + split["train_trend_future"] + split["train_seasonal_future"]
        )
        split["val_y_full"] = (
            split["val_y_resid"] + split["val_trend_future"] + split["val_seasonal_future"]
        )

        splits.append(split)
        start += step

    return splits