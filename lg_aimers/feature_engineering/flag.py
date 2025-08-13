# 해야함
def add_event_flag(df, window=28, std_q=0.70):
    minp = max(3, window//2)
    df['roll_std'] = df['residual'].rolling(window, min_periods=minp).std(ddof=0)
    s_hi = df['roll_std'].quantile(std_q)
    df['is_eventy_volatile'] = (df['roll_std'] >= s_hi).astype(int)
    return df

# 해야함
def add_outlier_flag(df, window=28, iqr_k=1.5):
    minp = max(3, window//2)
    q1 = df['residual'].rolling(window, min_periods=minp).quantile(0.25)
    q3 = df['residual'].rolling(window, min_periods=minp).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    df['is_outlier'] = ((df['residual'] < lower) | (df['residual'] > upper)).astype(int)
    return df