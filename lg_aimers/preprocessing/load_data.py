from preprocessing import (
    add_date_features, add_holiday_info,
    MultiLabelEncoder
)

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

    mle = MultiLabelEncoder() 
    mle.fit(df, ["store", "menu", "holiday"]) # 학습 데이터로 fit
    df = mle.transform(df)
    mle.save("encoders/label")

    # ---------------------------------
    # 추후 로드할 때
    mle2 = MultiLabelEncoder()
    mle2.load("encoders/label", ["store", "menu", "holiday"])

    # 로드한 걸로 새 데이터 변환
    df_test = mle2.transform(df_test)



    # 날짜 전처리
    df = add_date_features(df)
    df = add_holiday_info(df)
    df = df.drop(columns=['day', 'month', 'dow', 'week', 'day_of_year'])

    return df