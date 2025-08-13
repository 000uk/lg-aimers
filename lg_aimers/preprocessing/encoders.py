# preprocessing/encoders.py
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fit_label_encoders(df, store_col='store', menu_col='menu'):
    le_store = LabelEncoder().fit(df[store_col])
    le_menu  = LabelEncoder().fit(df[menu_col])
    return le_store, le_menu

def save_encoders(le_store, le_menu, store_path='le_store.pkl', menu_path='le_menu.pkl'):
    with open(store_path, 'wb') as f: pickle.dump(le_store, f)
    with open(menu_path, 'wb') as f: pickle.dump(le_menu, f)

def load_encoders(store_path='le_store.pkl', menu_path='le_menu.pkl'):
    with open(store_path, 'rb') as f: le_store = pickle.load(f)
    with open(menu_path, 'rb') as f: le_menu  = pickle.load(f)
    return le_store, le_menu

def encode_labels(df, le_store, le_menu):
    # 만약에 test에서 train에서 못 본 새로운 놈이 나오면..
    # UNK 인덱스를 주는 safe 버전 이것도 고려해봐야겟다
    df = df.copy()
    df['store_enc'] = le_store.transform(df['store'])
    df['menu_enc'] = le_menu.transform(df['menu'])
    return df