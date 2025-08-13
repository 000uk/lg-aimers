from statsmodels.tsa.seasonal import STL

def stl_decompose(df, target_col='sales_qty', group_col='store_menu', period=7):
    results = []

    for _, group in df.groupby(group_col):
        stl = STL(group[target_col], period=period, robust=True)
        res = stl.fit()
        group = group.copy()  # 원본 df 건드리지 않기 위해 복사
        group['trend'] = res.trend
        group['seasonal'] = res.seasonal
        group['residual'] = group[target_col] - (group['trend'] + group['seasonal'])
        results.append(group)

    return pd.concat(results, ignore_index=True)