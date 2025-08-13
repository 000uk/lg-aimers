import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={ # 영문으로 변환
    '영업일자': 'date',
    '영업장명_메뉴명': 'store_menu',
    '매출수량': 'sales_qty'
    })

    df['store'] = df['store_menu'].str.split('_').str[0]
    df['menu'] =  df['store_menu'].str.split('_').str[1]

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dow'] = df['date'].dt.weekday # 월=0, ..., 일=6
    df['week'] = df['date'].dt.isocalendar().week.astype(int)

    return df