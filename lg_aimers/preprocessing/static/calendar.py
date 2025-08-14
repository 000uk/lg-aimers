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

def add_holiday_info(df):
    years = df['year'].unique()
    kr_hols = holidays.KR(years=years)
    df['is_holiday'] = df['date'].dt.date.isin(kr_hols)
    df['holiday_name'] = df['date'].dt.date.map(lambda d: kr_hols.get(d, 'None'))
    return df
