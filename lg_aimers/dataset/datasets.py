import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from stl import extrapolate_trend, repeat_seasonal

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from stl import extrapolate_trend, repeat_seasonal

class WindowGenerator(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 input_len: int = 28,
                 pred_len: int = 7,
                 lag: int = 7,                 # 과거 k일
                 id_cols: tuple = ("store_menu_enc", "menu_enc"),
                 target_col: str = "residual",
                 used_feature_cols: list = None,  # 과거 구간 입력(Encoder)용
                 known_future_cols: list = None): # 미래 구간 입력(Decoder)용
        self.df = df.copy()
        self.input_len = input_len
        self.pred_len = pred_len
        self.lag = lag
        self.id_cols = id_cols
        self.target_col = target_col  # target_col=None 가능

        # 디폴트 feature 세팅
        if used_feature_cols is None:
            self.used_feature_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday",
                "is_holiday", "trend","seasonal"
            ]
        else:
            self.used_feature_cols = used_feature_cols

        if known_future_cols is None:
            self.known_future_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday",
                "is_holiday"
            ]
        else:
            self.known_future_cols = known_future_cols

        # 슬라이딩 윈도우 생성
        self.X_enc, self.X_dec_future, self.y_resid, \
            self.trend_future, self.seasonal_future, self.meta = self.build_windows()

    def build_windows(self):
        X_enc_list, X_dec_list, y_list, T_list, S_list = [], [], [], [], []
        meta_rows = []

        for (smid, mid), g in self.df.groupby(list(self.id_cols), sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            n = len(g)
            if n < self.input_len + self.pred_len:
                continue

            used_feats = g[self.used_feature_cols].values
            known_feats = g[self.known_future_cols].values
            if self.target_col is not None:
                resid = g[self.target_col].values
            trend = g["trend"].values
            seas = g["seasonal"].values
            sales_norm = g["sales_norm"].values
            dates = g["date"].values

            for t in range(self.input_len, n - self.pred_len + 1):
                enc_start = t - self.input_len
                enc_end = t
                dec_start = t
                dec_end = t + self.pred_len

                X_enc_window = used_feats[enc_start:enc_end, :].astype(np.float32)

                # lag features 생성
                lag_feats = np.zeros((self.input_len, self.lag), dtype=np.float32)
                for i in range(self.input_len):
                    start_idx = max(0, enc_start + i - self.lag)
                    end_idx = enc_start + i
                    lag_window = sales_norm[start_idx:end_idx]
                    if len(lag_window) > 0:
                        lag_feats[i, -len(lag_window):] = lag_window
                    else:
                        # 빈 배열일 경우에 대한 처리 (선택사항)
                        # 예: 0으로 채우기 등
                        lag_feats[i, :] = 0.0
                    # lag_feats[i, -len(lag_window):] = lag_window  # 부족한 경우 앞쪽 0 패딩
                
                # X_enc에 lag concat
                X_enc_window = np.concatenate([X_enc_window, lag_feats], axis=-1)

                X_enc_list.append(X_enc_window)
                X_dec_list.append(known_feats[dec_start:dec_end, :].astype(np.float32))

                if self.target_col is not None:
                    y_list.append(resid[dec_start:dec_end].astype(np.float32))

                # trend, seasonal 외삽
                # T_future = extrapolate_trend(trend[enc_start:enc_end], self.pred_len).astype(np.float32)
                # S_future = repeat_seasonal(seas[enc_start:enc_end], self.pred_len).astype(np.float32)
                # 걍 쓰지 말아보자
                T_future = np.zeros(self.pred_len, dtype=np.float32)
                S_future = np.zeros(self.pred_len, dtype=np.float32)

                T_list.append(T_future)
                S_list.append(S_future)

                meta_rows.append({
                    "store_menu_enc": smid,
                    "menu_enc": mid,
                    "start_date": pd.to_datetime(dates[enc_start]).date(),
                    "pred_start_date": pd.to_datetime(dates[dec_start]).date(),
                })

        X_enc = np.stack(X_enc_list)
        X_dec_future = np.stack(X_dec_list)
        if self.target_col is not None:
            y_resid = np.stack(y_list)
        else:
            y_resid = None
        trend_future = np.stack(T_list)
        seasonal_future = np.stack(S_list)
        meta = pd.DataFrame(meta_rows)

        return X_enc, X_dec_future, y_resid, trend_future, seasonal_future, meta

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        x_enc = torch.tensor(self.X_enc[idx])
        x_dec = torch.tensor(self.X_dec_future[idx])
        trend_future = torch.tensor(self.trend_future[idx])
        seasonal_future = torch.tensor(self.seasonal_future[idx])

        sample = {
            "X_enc": x_enc,
            "X_dec_future": x_dec,
            "trend_future": trend_future,
            "seasonal_future": seasonal_future,
            "store_menu_id": torch.tensor(self.meta["store_menu_enc"].iloc[idx], dtype=torch.long),
            "menu_id": torch.tensor(self.meta["menu_enc"].iloc[idx], dtype=torch.long),
        }

        # target_col이 있는 경우만 추가
        if self.target_col is not None:
            y_resid = torch.tensor(self.y_resid[idx])
            y_full = y_resid + trend_future + seasonal_future
            sample["y_resid"] = y_resid
            sample["y_full"] = y_full

        return sample
    
"""
- 과거 입력(X_enc)와 미래 입력(X_dec_future), 그리고 목표값(y_full)을 포함한 데이터셋을 PyTorch Dataset 형태로 준비.
- y_full = residual + trend + seasonal → 모델이 시계열 전체를 직접 예측하도록 함.
- Teacher Forcing을 위해 학습 시 디코더 입력에 이전 시점의 실제 y_full 값을 넣을 수 있게 설계.
- 모델에서 store_emb, menu_emb, store_menu_emb를 사용하기 위한 ID 값도 넣어줌
"""
class TSFullDataset(Dataset):
    def __init__(self, X_enc, X_dec_future, y_full,
                 store_menu_id=None, menu_id=None):
        self.X_enc = X_enc.astype(np.float32)
        self.X_dec_future = X_dec_future.astype(np.float32)
        self.y_full = y_full.astype(np.float32)
        self.store_menu_id = store_menu_id
        self.menu_id = menu_id

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return {
            "X_enc": torch.tensor(self.X_enc[idx], dtype=torch.float32),
            "X_dec_future": torch.tensor(self.X_dec_future[idx], dtype=torch.float32),
            "y_full": torch.tensor(self.y_full[idx], dtype=torch.float32),
            "store_menu_id": torch.tensor(self.store_menu_id[idx], dtype=torch.long),
            "menu_id": torch.tensor(self.menu_id[idx], dtype=torch.long),
        }

# 테스트용 데이터셋임. 추론할 떄 사용
class TSTestDataset(Dataset):
    def __init__(self, X_enc, X_dec_future, store_menu_id, menu_id):
        self.X_enc = X_enc
        self.X_dec_future = X_dec_future
        self.store_menu_id = store_menu_id
        self.menu_id = menu_id

    def __len__(self):
        return len(self.X_enc)

    def __getitem__(self, idx):
        return {
            "X_enc": torch.tensor(self.X_enc[idx], dtype=torch.float),
            "X_dec_future": torch.tensor(self.X_dec_future[idx], dtype=torch.float),
            "store_menu_id": torch.tensor(self.store_menu_id[idx], dtype=torch.long),
            "menu_id": torch.tensor(self.menu_id[idx], dtype=torch.long),
        }