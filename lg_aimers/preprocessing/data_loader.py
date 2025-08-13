import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


    # 원핫 인코딩
    df = pd.get_dummies(df, columns=['store', 'menu'], prefix=['store', 'menu'])

    # 라벨 인코딩(임베딩용)
    le_store = LabelEncoder()
    df['store_enc'] = le_store.fit_transform(df['store_menu'].str.split('_').str[0])

    le_menu = LabelEncoder()
    df['menu_enc'] = le_menu.fit_transform(df['store_menu'].str.split('_').str[1])

    return df, le_store, le_menu