from statsmodels.tsa.seasonal import STL
import pandas as pd

def stl_decompose(df):
    results = []

    for _, group in df.groupby('store_menu'):
        stl = STL(group['sales_qty'], period=7, robust=True) # 주간 패턴이 아무래도 중요할 거 같아서 7로 함
        res = stl.fit()
        group = group.copy()  # 원본 df 건드리지 않기 위해 복사
        group['trend'] = res.trend
        group['seasonal'] = res.seasonal
        group['residual'] = group['sales_qty'] - (group['trend'] + group['seasonal'])
        results.append(group)

    return pd.concat(results, ignore_index=True)