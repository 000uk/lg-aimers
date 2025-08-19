import numpy as np
from typing import Dict, Tuple, List

# -----------------------------
# 3) 슬라이딩 윈도우 데이터셋 만들기 (Residual-only)
# -----------------------------
def build_windows_per_series(
    df: pd.DataFrame,
    input_len: int = 28,
    pred_len: int = 7,
    id_cols: Tuple[str, str] = ("store_enc", "menu_enc"),
    date_col: str = "date",
    target_col: str = "sales_qty",
    trend_col: str = "trend",
    seasonal_col: str = "seasonal",
    residual_col: str = "residual",
    known_future_cols: List[str] = None,
    used_feature_cols: List[str] = None,
) -> Dict[str, np.ndarray]:
    """
    시계열을 (store_enc, menu_enc)별로 분리하여
    - X_enc: (N, input_len, num_features)
    - X_dec_future: (N, pred_len, num_features_known_future)  ← patchTFT decoder covariates
    - y_resid: (N, pred_len)  ← 모델이 학습할 타깃(잔차)
    - trend_future: (N, pred_len)
    - seasonal_future: (N, pred_len)
    - meta: 각 윈도우에 대응하는 (store_enc, menu_enc, start_date)
    를 생성.
    """
    if known_future_cols is None:
        # 캘린더/휴일 기반 known-in-advance 피처들 넣어주면 좋아
        known_future_cols = [
            "dow", "month", "day_sin", "day_cos",
            "holiday_enc", "offblock_len", "offblock_pos",
            "days_to_next_holiday", "days_since_prev_holiday"
        ]

    if used_feature_cols is None:
        # 인코더 입력(과거 구간)에 사용할 피처들
        used_feature_cols = [
            "dow", "month", "day_sin", "day_cos",
            "holiday_enc", "offblock_len", "offblock_pos",
            "days_to_next_holiday", "days_since_prev_holiday",
            # 필요시 trend/seasonal 과거값을 힌트로 추가
            "trend", "seasonal"
        ]

    X_enc_list, X_decf_list, Y_resid_list = [], [], []
    T_future_list, S_future_list = [], []
    meta_rows = []

    # 시계열을 id별로 나눔
    for (sid, mid), g in df.groupby(list(id_cols), sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)
        n = len(g)

        # 최소 길이 체크
        min_len = input_len + pred_len
        if n < min_len:
            continue

        # 넘파이로 뽑아두면 빠름
        used_feats = g[used_feature_cols].values
        known_future_feats = g[known_future_cols].values
        resid = g[residual_col].values
        trend = g[trend_col].values
        seas = g[seasonal_col].values
        dates = g[date_col].values

        # 슬라이딩: [t-input_len, ..., t-1] → [t, ..., t+pred_len-1]
        # t는 input_len부터 시작해서 n - pred_len까지 가능
        for t in range(input_len, n - pred_len + 1):
            enc_start = t - input_len
            enc_end = t              # not inclusive
            dec_start = t
            dec_end = t + pred_len   # not inclusive

            X_enc = used_feats[enc_start:enc_end, :]                 # (input_len, F)
            X_dec_future = known_future_feats[dec_start:dec_end, :]  # (pred_len, Ff)

            y_resid = resid[dec_start:dec_end]                       # (pred_len,)

            last_trend = trend[enc_start:enc_end]
            last_seas = seas[enc_start:enc_end]

            trend_future = extrapolate_trend_linear(last_trend, pred_len)      # (pred_len,)
            seas_future = repeat_seasonal_weekly(last_seas, pred_len)          # (pred_len,)

            X_enc_list.append(X_enc.astype(np.float32))
            X_decf_list.append(X_dec_future.astype(np.float32))
            Y_resid_list.append(y_resid.astype(np.float32))
            T_future_list.append(trend_future.astype(np.float32))
            S_future_list.append(seas_future.astype(np.float32))

            meta_rows.append({
                "store_enc": sid,
                "menu_enc": mid,
                "start_date": pd.to_datetime(dates[enc_start]).date(),
                "pred_start_date": pd.to_datetime(dates[dec_start]).date()
            })

    # 스택
    X_enc = np.stack(X_enc_list) if X_enc_list else np.empty((0, input_len, len(used_feature_cols)), dtype=np.float32)
    X_dec_future = np.stack(X_decf_list) if X_decf_list else np.empty((0, pred_len, len(known_future_cols)), dtype=np.float32)
    y_resid = np.stack(Y_resid_list) if Y_resid_list else np.empty((0, pred_len), dtype=np.float32)
    trend_future = np.stack(T_future_list) if T_future_list else np.empty((0, pred_len), dtype=np.float32)
    seasonal_future = np.stack(S_future_list) if S_future_list else np.empty((0, pred_len), dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    info = {
        "used_feature_cols": used_feature_cols,
        "known_future_cols": known_future_cols,
        "input_len": input_len,
        "pred_len": pred_len,
    }
    return {
        "X_enc": X_enc,
        "X_dec_future": X_dec_future,
        "y_resid": y_resid,
        "trend_future": trend_future,
        "seasonal_future": seasonal_future,
        "meta": meta,
        "info": info
    }