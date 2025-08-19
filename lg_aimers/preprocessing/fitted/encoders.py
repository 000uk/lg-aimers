# preprocessing/encoders.py
import pickle
from sklearn.preprocessing import LabelEncoder

def fit_label_encoders(df):
    le_store    = LabelEncoder().fit(df["store"])
    le_menu     = LabelEncoder().fit(df["menu"])
    le_holiday  = LabelEncoder().fit(df["holiday"])
    return le_store, le_menu, le_holiday

def save_encoders(le_store, le_menu, le_holiday, 
                  store_path='le_store.pkl', menu_path='le_menu.pkl', holiday_path = 'le_holiday.pkl'):
    with open(store_path, 'wb') as f: pickle.dump(le_store, f)
    with open(menu_path, 'wb') as f: pickle.dump(le_menu, f)
    with open(holiday_path, 'wb') as f: pickle.dump(le_holiday, f)

def load_encoders(store_path='le_store.pkl', menu_path='le_menu.pkl', holiday_path = 'le_holiday.pkl'):
    with open(store_path, 'rb') as f: le_store = pickle.load(f)
    with open(menu_path, 'rb') as f: le_menu  = pickle.load(f)
    with open(holiday_path, 'rb') as f: le_holiday  = pickle.load(f)
    return le_store, le_menu, le_holiday

def encode_labels(df, le_store, le_menu, le_holiday):
    # 만약에 test에서 train에서 못 본 새로운 놈이 나오면..
    # UNK 인덱스를 주는 safe 버전 이것도 고려해봐야겟다
    df = df.copy()
    df['store_enc'] = le_store.transform(df['store'])
    df['menu_enc'] = le_menu.transform(df['menu'])
    df['holiday_enc'] = le_holiday.transform(df['holiday'])
    return df

# class MultiLabelEncoder:
#     def __init__(self):
#         self.encoders = {}
#         self.unk_index = {}

#     def fit(self, df, columns):
#         for col in columns:
#             le = LabelEncoder()
#             le.fit(df[col])
#             self.encoders[col] = le
#             self.unk_index[col] = len(le.classes_)  # UNK index

#     def transform(self, df):
#         df = df.copy()
#         for col, le in self.encoders.items():
#             df[col + "_enc"] = df[col].apply(
#                 lambda x: le.transform([x])[0] if x in le.classes_ else self.unk_index[col]
#             )
#         return df

#     def save(self, path_prefix):
#         for col, le in self.encoders.items():
#             with open(f"{path_prefix}_{col}.pkl", "wb") as f:
#                 pickle.dump(le, f)

#     def load(self, path_prefix, columns):
#         for col in columns:
#             with open(f"{path_prefix}_{col}.pkl", "rb") as f:
#                 self.encoders[col] = pickle.load(f)
#                 self.unk_index[col] = len(self.encoders[col].classes_)