def load_data(df):
    df = df.copy()

    # 컬럼명 영문 변환
    df = df.rename(columns={
    '영업일자': 'date',
    '영업장명_메뉴명': 'store_menu',
    '매출수량': 'sales_qty'
    })

    # store/menu 분리
    df['store'] = df['store_menu'].str.split('_').str[0]
    df['menu'] =  df['store_menu'].str.split('_').str[1]

    # transform: 원본 DataFrame 크기와 맞춰 각 행에 정규화 값 할당
    df['sales_norm'] = df.groupby('store_menu')['sales_qty'].transform(
        lambda x: (x - x.mean()) / x.std()
    ).fillna(0) # x.std()가 0이면 NaN이 나옴
    
    return df