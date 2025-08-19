def load_data(df):
    df = df.copy()

    # 컬럼명 영문 변환
    df = df.rename(columns={
    '영업일자': 'date',
    '영업장명_메뉴명': 'store_menu',
    '매출수량': 'sales_qty'
    })

    # store/menu 분리 및 전처리
    df['store'] = df['store_menu'].str.split('_').str[0]
    df['menu'] =  df['store_menu'].str.split('_').str[1]

    return df