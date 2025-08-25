# preprocessing/encoders.py
import pickle
from sklearn.preprocessing import LabelEncoder

# preprocessing/encoders.py
import pickle
from sklearn.preprocessing import LabelEncoder

# preprocessing/encoders.py
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fit_label_encoders(df):
    le_store_menu = LabelEncoder().fit(df["store_menu"])
    le_store      = LabelEncoder().fit(df["store"])
    le_menu       = LabelEncoder().fit(df["menu"])
    le_holiday    = LabelEncoder().fit(df["holiday"])
    return le_store_menu, le_store, le_menu, le_holiday

def save_encoders(le_store_menu, le_store, le_menu, le_holiday, 
                  store_menu_path='le_store_menu.pkl', store_path='le_store.pkl', 
                  menu_path='le_menu.pkl', holiday_path='le_holiday.pkl'):
    with open(store_menu_path, 'wb') as f: pickle.dump(le_store_menu, f)
    with open(store_path, 'wb') as f: pickle.dump(le_store, f)
    with open(menu_path, 'wb') as f: pickle.dump(le_menu, f)
    with open(holiday_path, 'wb') as f: pickle.dump(le_holiday, f)

def load_encoders(store_menu_path='le_store_menu.pkl', store_path='le_store.pkl', 
                  menu_path='le_menu.pkl', holiday_path='le_holiday.pkl'):
    with open(store_menu_path, 'rb') as f: le_store_menu = pickle.load(f)              
    with open(store_path, 'rb') as f: le_store = pickle.load(f)
    with open(menu_path, 'rb') as f: le_menu  = pickle.load(f)
    with open(holiday_path, 'rb') as f: le_holiday  = pickle.load(f)
    return le_store_menu, le_store, le_menu, le_holiday

def encode_labels(df, le_store_menu, le_store, le_menu, le_holiday):
    df = df.copy()
    df['store_menu_enc'] = np.where(df['store_menu'].isin(le_store_menu.classes_),
                                    le_store_menu.transform(df['store_menu']), -1)
    df['store_enc'] = np.where(df['store'].isin(le_store.classes_),
                               le_store.transform(df['store']), -1)
    df['menu_enc'] = np.where(df['menu'].isin(le_menu.classes_),
                              le_menu.transform(df['menu']), -1)
    df['holiday_enc'] = np.where(df['holiday'].isin(le_holiday.classes_),
                                 le_holiday.transform(df['holiday']), -1)
    return df