import pandas as pd
import numpy as np
import holidays

def add_date_features(df):
    # 날짜 컬럼 처리
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dow'] = df['date'].dt.weekday # 월=0, ..., 일=6
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear

    # 연중일 sin/cos
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df

"""
한국 공휴일(holidays.KR) + 주말을 이용해 known-in-advance 이벤트 피처 생성.
- offblock_len: 연휴(주말+공휴일 포함) 길이
- offblock_pos: 연휴 내 0-based 위치 (첫날=0)
- days_to_next_holiday / days_since_prev_holiday: 최근/다음 공휴일까지 일수
"""
def add_holiday_info(df):
    df = df.copy()

    # 공휴일 캘린더 준비 (데이터 경계 여유 잡기)
    years = pd.RangeIndex(df["date"].dt.year.min() - 1,
                          df["date"].dt.year.max() + 2)
    kr_hols = holidays.KR(years=list(years))
    
    df['holiday_name'] = df['date'].dt.date.map(lambda d: kr_hols.get(d, 'None'))

    # 날짜 단위 인덱스 테이블 만들기 (중복 방지)
    cal = pd.DataFrame({"date": pd.to_datetime(sorted(df["date"].unique()))})

    # 주말/휴일·연휴 블록
    cal["is_holiday"] = (df["holiday_name"] != "None").astype(int)
    cal["is_weekend"] = (df["dow"] >= 5).astype(int)
    cal["is_offday"] = ((cal["is_holiday"] == 1) | (cal["is_weekend"] == 1)).astype(int)

    # 연휴(주말+공휴일) 연속 블록 계산
    cal = cal.sort_values("date").reset_index(drop=True)
    day_diff = cal["date"].diff().dt.days.fillna(1)
    # offday가 시작되는 지점(True)만 누적합 → block id
    block_start = (cal["is_offday"].eq(1) &
                   (cal["is_offday"].shift(fill_value=0).eq(0) | (day_diff != 1)))
    cal["offblock_id"] = np.where(cal["is_offday"].eq(1), block_start.cumsum(), np.nan)
    cal["offblock_len"] = cal.groupby("offblock_id")["is_offday"].transform("count")
    cal["offblock_pos"] = cal.groupby("offblock_id").cumcount()

    # 최근/다음 공휴일까지 일수
    # 비휴일(0)에 대해 최근/다음 휴일(1)까지 거리 계산
    cal["is_holiday_int"] = cal["is_holiday"] # For ffill/bfill
    cal["days_since_prev_holiday"] = cal.loc[cal["is_holiday_int"].eq(1), "date"].reindex(cal["date"].index, method="ffill")
    cal["days_since_prev_holiday"] = (cal["date"] - cal["days_since_prev_holiday"]).dt.days.fillna(1000)
    cal["days_to_next_holiday"] = cal.loc[cal["is_holiday_int"].eq(1), "date"].reindex(cal["date"].index, method="bfill")
    cal["days_to_next_holiday"] = (cal["days_to_next_holiday"] - cal["date"]).dt.days.fillna(1000)
    cal = cal.drop(columns="is_holiday_int")

    # 원본 테이블에 merge
    to_merge = cal[["date", "offblock_len", "offblock_pos",
                "days_to_next_holiday", "days_since_prev_holiday"]]
    df = df.merge(to_merge, on="date", how="left")

    return df
