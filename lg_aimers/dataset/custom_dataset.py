import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Custom_Dataset(Dataset):
    """
    Transformer 학습용 시계열 Dataset
    - input_len: 과거 관측 길이
    - pred_len: 예측 길이
    - df: preprocess 완료된 DataFrame
    """

    def __init__(self, 
                 df: pd.DataFrame,
                 input_len: int = 28,
                 pred_len: int = 7,
                 id_cols: tuple = ("store_enc", "menu_enc"),
                 target_col: str = "residual",
                 used_feature_cols: list = None,
                 known_future_cols: list = None):
        self.df = df.copy()
        self.input_len = input_len
        self.pred_len = pred_len
        self.id_cols = id_cols
        self.target_col = target_col

        # 디폴트 feature
        if used_feature_cols is None:
            self.used_feature_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday",
                "trend","seasonal"
            ]
        else:
            self.used_feature_cols = used_feature_cols

        if known_future_cols is None:
            self.known_future_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday"
            ]
        else:
            self.known_future_cols = known_future_cols

        # 슬라이딩 윈도우 생성
        self.X_enc, self.X_dec_future, self.y_resid, self.trend_future, self.seasonal_future, self.meta = \
            self.build_windows()

    def build_windows(self):
        X_enc_list, X_dec_list, y_list, T_list, S_list = [], [], [], [], []
        meta_rows = []

        for (sid, mid), g in self.df.groupby(list(self.id_cols), sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            n = len(g)
            if n < self.input_len + self.pred_len:
                continue

            used_feats = g[self.used_feature_cols].values
            known_feats = g[self.known_future_cols].values
            resid = g[self.target_col].values
            trend = g["trend"].values
            seas = g["seasonal"].values
            dates = g["date"].values

            for t in range(self.input_len, n - self.pred_len + 1):
                enc_start = t - self.input_len
                enc_end = t
                dec_start = t
                dec_end = t + self.pred_len

                X_enc_list.append(used_feats[enc_start:enc_end, :].astype(np.float32))
                X_dec_list.append(known_feats[dec_start:dec_end, :].astype(np.float32))
                y_list.append(resid[dec_start:dec_end].astype(np.float32))
                # trend/seasonal future는 필요하면 외삽 함수로 대체 가능
                T_list.append(trend[dec_start:dec_end].astype(np.float32))
                S_list.append(seas[dec_start:dec_end].astype(np.float32))

                meta_rows.append({
                    "store_enc": sid,
                    "menu_enc": mid,
                    "start_date": pd.to_datetime(dates[enc_start]).date(),
                    "pred_start_date": pd.to_datetime(dates[dec_start]).date()
                })

        # 스택
        X_enc = np.stack(X_enc_list) if X_enc_list else np.empty((0, self.input_len, len(self.used_feature_cols)), dtype=np.float32)
        X_dec_future = np.stack(X_dec_list) if X_dec_list else np.empty((0, self.pred_len, len(self.known_future_cols)), dtype=np.float32)
        y_resid = np.stack(y_list) if y_list else np.empty((0, self.pred_len), dtype=np.float32)
        trend_future = np.stack(T_list) if T_list else np.empty((0, self.pred_len), dtype=np.float32)
        seasonal_future = np.stack(S_list) if S_list else np.empty((0, self.pred_len), dtype=np.float32)
        meta = pd.DataFrame(meta_rows)

        return X_enc, X_dec_future, y_resid, trend_future, seasonal_future, meta

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return {
            "X_enc": torch.tensor(self.X_enc[idx]),
            "X_dec_future": torch.tensor(self.X_dec_future[idx]),
            "y_resid": torch.tensor(self.y_resid[idx]),
            "trend_future": torch.tensor(self.trend_future[idx]),
            "seasonal_future": torch.tensor(self.seasonal_future[idx])
        }
