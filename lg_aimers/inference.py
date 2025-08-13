from lg_aimers.feature_engineering.flag import add_eventy_flag, add_outlier_flag
from preprocessing import load_data, fit_label_encoders, load_encoders, encode_labels
DATA_PATH = "data/test/TEST_00.csv"

df_test = data_preprocess(df_test)

# 인코딩
le_store, le_menu = load_encoders('le_store.pkl', 'le_menu.pkl')
df_test, meta = encode_with_encoders(df_test, le_store, le_menu)

# 모델 입력 X만 따로 생성 (drop)
X_test = df_test.drop(columns=['date', 'store_menu'])  

df = stl_decompose(df, target_col='sales_qty', period=7)
df = add_eventy_flag(df, window=28)
df = add_outlier_flag(df, window=28)


# 예측
preds = model.predict(X_test)

# 제출용: 원본 정보 + 예측 결과
submission = df_test[['date', 'store_menu']].copy()
submission['predicted_sales'] = preds
submission.to_csv('submission.csv', index=False)


# 추론용 df_test
df_submit = df_test[['store_menu']].copy()
df_submit['pred'] = model.predict(X_test)
df_submit.to_csv('submission.csv', index=False)