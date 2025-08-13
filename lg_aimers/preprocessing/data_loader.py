import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(df):
    # 컬럼명 영문 변환
    df = df.rename(columns={
    '영업일자': 'date',
    '영업장명_메뉴명': 'store_menu',
    '매출수량': 'sales_qty'
    })

    # store/menu 분리
    df['store'] = df['store_menu'].str.split('_').str[0]
    df['menu'] =  df['store_menu'].str.split('_').str[1]

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