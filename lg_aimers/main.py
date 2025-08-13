from preprocess.feature_engineering import add_eventy_flag, add_outlier_flag
import pandas as pd
DATA_PATH = "data/train/train.csv"

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

df = load_data("data/train.csv")
df = stl_decompose(df, target_col='sales_qty', period=7)
df = add_eventy_flag(df, window=28)
df = add_outlier_flag(df, window=28)