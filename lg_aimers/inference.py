from lg_aimers.feature_engineering.flag import add_eventy_flag, add_outlier_flag
DATA_PATH = "data/train.csv"

'''
le_store = LabelEncoder()
df['store_enc'] = le_store.fit_transform(df['store_menu'].str.split('_').str[0])
fit: store 컬럼에 있는 고유값을 확인하고, 0,1,2,... 숫자에 매핑

transform: 원래 값들을 위 숫자로 바꿈

한 줄로 합쳐져서 학습 데이터에 바로 적용 가능하다는 의미죠.

💡 중요한 점:

학습 데이터에서는 fit_transform()
테스트/추론 데이터에서는 이미 학습한 encoders로 transform()만 해야 함

df_test['store_enc'] = le_store.transform(df_test['store'])
'''


df_test = pd.read_csv('test.csv')  # 아직 인코딩 안 된 상태

# 학습 때 저장한 encoder 불러오기
with open('le_store.pkl', 'rb') as f:
    le_store = pickle.load(f)
with open('le_menu.pkl', 'rb') as f:
    le_menu = pickle.load(f)

# 테스트 데이터에 transform 적용
df_test['store_enc'] = le_store.transform(df_test['store'])
df_test['menu_enc'] = le_menu.transform(df_test['menu'])

df = load_data("data/train.csv")
df = stl_decompose(df, target_col='sales_qty', period=7)
df = add_eventy_flag(df, window=28)
df = add_outlier_flag(df, window=28)