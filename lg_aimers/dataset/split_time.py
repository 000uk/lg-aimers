# -----------------------------
# 4) 학습/검증 분리 (간단 rolling or time split)
# -----------------------------
def time_based_split(
    windows: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    val_ratio: float = 0.1
):
    """
    meta['pred_start_date'] 기준으로 시간순 정렬 후 뒤쪽 val_ratio를 검증으로 사용.
    """
    order = np.argsort(meta['pred_start_date'].values)
    for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future"]:
        windows[k] = windows[k][order]
    meta = meta.iloc[order].reset_index(drop=True)

    n = len(meta)
    n_val = max(1, int(np.round(n * val_ratio)))

    train_slice = slice(0, n - n_val)
    val_slice = slice(n - n_val, n)

    split = {}
    for k in ["X_enc", "X_dec_future", "y_resid", "trend_future", "seasonal_future"]:
        split[f"train_{k}"] = windows[k][train_slice]
        split[f"val_{k}"] = windows[k][val_slice]
    split["train_meta"] = meta.iloc[0:n - n_val].reset_index(drop=True)
    split["val_meta"] = meta.iloc[n - n_val:].reset_index(drop=True)
    split["info"] = windows["info"]
    return split