import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from stl import extrapolate_trend, repeat_seasonal

class Custom_Dataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 input_len: int = 28,
                 pred_len: int = 7,
                 id_cols: tuple = ("store_enc", "menu_enc"),
                 target_col: str = "residual",
                 used_feature_cols: list = None, # 과거 구간 입력(Encoder)용
                 known_future_cols: list = None): # 미래 구간 입력(Decoder)용
        self.df = df.copy()
        self.input_len = input_len
        self.pred_len = pred_len
        self.id_cols = id_cols
        self.target_col = target_col

        # 디폴트 feature 세팅
        if used_feature_cols is None:
            self.used_feature_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday",
                "is_holiday","is_weekend","is_offday",
                "trend","seasonal"
            ]
        else:
            self.used_feature_cols = used_feature_cols

        if known_future_cols is None:
            self.known_future_cols = [
                "day_sin","day_cos","month_sin","month_cos","dow_sin","dow_cos",
                "holiday_enc","offblock_len","offblock_pos",
                "days_to_next_holiday","days_since_prev_holiday",
                "is_holiday","is_weekend","is_offday"
            ]
        else:
            self.known_future_cols = known_future_cols

        # 슬라이딩 윈도우 생성.. 여기서 실제로 데이터 자르고 붙인대
        """
        self.X_enc: 인코더 입력값 (과거 시계열 데이터일 가능성 있음)
        self.X_dec_future: 디코더가 사용할 미래 입력값
        self.y_resid: 잔차(residuals), 즉 예측값과 실제값의 차이
        self.trend_future: 미래 트렌드 성분
        self.seasonal_future: 미래 계절성 성분
        self.meta: 메타데이터 (예: 타임스탬프, 범주형 정보 등)
        """
        self.X_enc, self.X_dec_future, self.y_resid, \
            self.trend_future, self.seasonal_future, self.meta = self.build_windows()

    def build_windows(self):
        X_enc_list, X_dec_list, y_list, T_list, S_list = [], [], [], [], []
        meta_rows = []

        # store랑 menu로 그룹화해서 윈도우 나눔
        for (sid, mid), g in self.df.groupby(list(self.id_cols), sort=False):
            g = g.sort_values("date").reset_index(drop=True) # date로 정렬
            n = len(g)
            if n < self.input_len + self.pred_len:
                continue

            # 필요한 feature 및 target 값들을 numpy array로 뽑음
            used_feats = g[self.used_feature_cols].values
            known_feats = g[self.known_future_cols].values
            resid = g[self.target_col].values
            trend = g["trend"].values
            seas = g["seasonal"].values
            dates = g["date"].values

            # t는 윈도우의 현재 시점(입력28일과 출력7일의 사이)
            for t in range(self.input_len, n - self.pred_len + 1):
                enc_start = t - self.input_len
                enc_end = t
                dec_start = t
                dec_end = t + self.pred_len

                X_enc_list.append(used_feats[enc_start:enc_end, :].astype(np.float32))
                X_dec_list.append(known_feats[dec_start:dec_end, :].astype(np.float32))
                y_list.append(resid[dec_start:dec_end].astype(np.float32))
                
                # trend, season 외삽
                last_trend = trend[enc_start:enc_end]   # 과거 trend 구간
                last_seasonal = seas[enc_start:enc_end] # 과거 seasonal 구간

                T_future = extrapolate_trend(last_trend, self.pred_len).astype(np.float32)
                S_future = repeat_seasonal(last_seasonal, self.pred_len).astype(np.float32)

                T_list.append(T_future)
                S_list.append(S_future)

                # 어떤 매장/메뉴/기간에서 뽑힌 샘플인지 추적하기 위한 메타정보
                meta_rows.append({
                    "store_enc": sid,
                    "menu_enc": mid,
                    "start_date": pd.to_datetime(dates[enc_start]).date(),
                    "pred_start_date": pd.to_datetime(dates[dec_start]).date()
                })

        # 스택(numpy에서 tensor로 변환하기 전에 배열로 모아둠)
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
        y_resid = torch.tensor(self.y_resid[idx])          # 모델이 학습할 타깃
        trend_future = torch.tensor(self.trend_future[idx]) 
        seasonal_future = torch.tensor(self.seasonal_future[idx])

        y_full = y_resid + trend_future + seasonal_future
        return {
            "X_enc": torch.tensor(self.X_enc[idx]),
            "X_dec_future": torch.tensor(self.X_dec_future[idx]),
            #"y_resid": torch.tensor(self.y_resid[idx]),
            #"trend_future": torch.tensor(self.trend_future[idx]),
            #"seasonal_future": torch.tensor(self.seasonal_future[idx])
            "y_resid": y_resid,
            "trend_future": trend_future,
            "seasonal_future": seasonal_future,
            "y_full": y_full
        }
