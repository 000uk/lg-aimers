import pandas as pd
from preprocessing import load_data, MultiLabelEncoder, add_date_features, add_holiday_info, 
from dataset.custom_dataset import TransformerTimeSeriesDataset
from torch.utils.data import DataLoader

DATA_PATH = "data/train/train.csv"

df_train = pd.read_csv(DATA_PATH)
df_train = load_data(df_train)

df_train = add_date_features(df_train)
df_train = add_holiday_info(df_train)
df_train = df_train.drop(columns=['day', 'month', 'dow', 'week', 'day_of_year'])

mle = MultiLabelEncoder() 
mle.fit(df_train, ["store", "menu", "holiday"]) # 학습 데이터로 fit
df = mle.transform(df_train)
mle.save("encoders/label")
# mle2 = MultiLabelEncoder()
# mle2.load("encoders/label", ["store", "menu", "holiday"])
# df_test = mle2.transform(df_test)

'''
# Dataset 생성
train_dataset = TransformerTimeSeriesDataset(train_df, input_len=28, pred_len=7)
val_dataset   = TransformerTimeSeriesDataset(val_df, input_len=28, pred_len=7)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)



df_train = stl_decompose(df_train)
'''
