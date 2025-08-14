# feature_engineering/events.py

import pandas as pd
import numpy as np
import holidays

def add_kr_holiday_features(df: pd.DataFrame, date_col: str = "date"):
    """
    한국 공휴일(holidays.KR) + 주말을 이용해 known-in-advance 이벤트 피처 생성.
    - is_holiday: 공휴일 여부 (법정 공휴일/대체공휴일 포함)
    - holiday_name: 공휴일 이름(없으면 'None')
    - is_weekend: 토/일
    - is_offday: 공휴일 or 주말
    - offblock_len: 연휴(주말+공휴일 포함) 길이
    - offblock_pos: 연휴 내 0-based 위치 (첫날=0)
    - is_seollal_period: 설 연휴(전날/당일/다음날/대체 포함) 여부
    - is_chuseok_period: 추석 연휴(전날/당일/다음날/대체 포함) 여부
    - is_eve_day: (전날) 표기 포함 휴일 플래그
    - is_following_day: (다음날) 표기 포함 휴일 플래그
    - is_substitute_holiday: 대체공휴일 여부
    - days_to_next_holiday / days_since_prev_holiday: 최근/다음 공휴일까지 일수
    """
    df = df.copy()
    # 날짜 dtype 보정
    df[date_col] = pd.to_datetime(df[date_col])

    # 공휴일 캘린더 준비 (데이터 경계 여유 잡기)
    years = pd.RangeIndex(df[date_col].dt.year.min() - 1,
                          df[date_col].dt.year.max() + 2)
    kr_hols = holidays.KR(years=list(years))

    # 날짜 단위 인덱스 테이블 만들기 (중복 방지)
    cal = pd.DataFrame({date_col: pd.to_datetime(sorted(df[date_col].unique()))})
    cal["date_only"] = cal[date_col].dt.date

    # 공휴일 이름 / 플래그
    def _get_hol_name(d):
        name = kr_hols.get(d, None)
        return name if name is not None else "None"

    cal["holiday_name"] = cal["date_only"].apply(_get_hol_name)
    cal["is_holiday"] = (cal["holiday_name"] != "None").astype(int)

    # 주말/휴일·연휴 블록
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["is_offday"] = ((cal["is_holiday"] == 1) | (cal["is_weekend"] == 1)).astype(int)

    # 연휴(주말+공휴일) 연속 블록 계산
    cal = cal.sort_values(date_col).reset_index(drop=True)
    day_diff = cal[date_col].diff().dt.days.fillna(1)
    # offday가 시작되는 지점(True)만 누적합 → block id
    block_start = (cal["is_offday"].eq(1) &
                   (cal["is_offday"].shift(fill_value=0).eq(0) | (day_diff != 1)))
    cal["offblock_id"] = np.where(cal["is_offday"].eq(1), block_start.cumsum(), np.nan)

    # 블록 길이/포지션
    cal["offblock_len"] = (
        cal.groupby("offblock_id")["is_offday"].transform("size")
        .where(cal["is_offday"].eq(1), 0)
    ).astype(int)
    cal["offblock_pos"] = (
        cal.groupby("offblock_id").cumcount().where(cal["is_offday"].eq(1), 0)
    ).astype(int)

    # 설/추석/전날/다음날/대체 식별 (영문/국문 명칭 모두 대응)
    def name_flags(name: str):
        n = name.lower()
        is_sub = ("대체" in name) or ("substitute" in n)
        is_eve = ("전날" in name) or ("day preceding" in n) or ("eve" in n)
        is_follow = ("다음날" in name) or ("day following" in n)
        is_seol = ("설" in name) or ("seollal" in n)
        is_chu = ("추석" in name) or ("chuseok" in n)
        # 설/추석 기간: 이름에 설/추석이 들어가면 전날/당일/다음날/대체 포함
        is_seol_period = is_seol
        is_chu_period = is_chu
        return pd.Series({
            "is_substitute_holiday": int(is_sub),
            "is_eve_day": int(is_eve),
            "is_following_day": int(is_follow),
            "is_seollal_period": int(is_seol_period),
            "is_chuseok_period": int(is_chu_period),
        })

    cal = pd.concat([cal, cal["holiday_name"].apply(name_flags)], axis=1)

    # 가까운 공휴일까지 거리 (평일에도 계산 → known in advance)
    hol_dates = pd.to_datetime([d for d in kr_hols.keys()])
    hol_dates = np.array(sorted(hol_dates))

    def days_to_next(d):
        # d보다 크거나 같은 첫 공휴일까지 거리
        idx = hol_dates.searchsorted(d, side="left")
        if idx >= len(hol_dates):  # 없으면 NaN
            return np.nan
        return int((hol_dates[idx] - d).days)

    def days_since_prev(d):
        idx = hol_dates.searchsorted(d, side="right") - 1
        if idx < 0:
            return np.nan
        return int((d - hol_dates[idx]).days)

    cal_ts = pd.to_datetime(cal["date_only"])
    cal["days_to_next_holiday"] = cal_ts.apply(days_to_next)
    cal["days_since_prev_holiday"] = cal_ts.apply(days_since_prev)

    # 최종 병합 (date만으로 merge)
    to_merge = cal[[date_col, "is_holiday", "holiday_name",
                    "is_weekend", "is_offday", "offblock_len", "offblock_pos",
                    "is_seollal_period", "is_chuseok_period",
                    "is_eve_day", "is_following_day",
                    "is_substitute_holiday",
                    "days_to_next_holiday", "days_since_prev_holiday"]]
    df = df.merge(to_merge, on=date_col, how="left")

    # 타입/결측 보정
    int_cols = ["is_holiday", "is_weekend", "is_offday",
                "offblock_len", "offblock_pos",
                "is_seollal_period", "is_chuseok_period",
                "is_eve_day", "is_following_day", "is_substitute_holiday"]
    for c in int_cols:
        df[c] = df[c].fillna(0).astype(int)
    df["holiday_name"] = df["holiday_name"].fillna("None")

    return df