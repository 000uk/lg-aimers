from preprocessing import (
    add_date_features, add_holiday_info,
    fit_label_encoders, save_encoders, encode_labels
)

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

    df_train = add_date_features(df_train)
    df_train = add_holiday_info(df_train)
    df_train = df_train.drop(columns=['day', 'month', 'dow', 'week', 'day_of_year'])

    le_store, le_menu, le_holiday = fit_label_encoders(df_train)
    save_encoders(le_store, le_menu, le_holiday, 'le_store.pkl', 'le_menu.pkl', 'le_holiday.pkl')
    df_train = encode_labels(df_train, le_store, le_menu, le_holiday)

    return df