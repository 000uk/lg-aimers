import numpy as np

def extrapolate_trend_linear(last_trend: np.ndarray, horizon: int) -> np.ndarray:
    """
    마지막 구간 trend를 1차 선형으로 완만하게 외삽.
    last_trend: (L,)  최근 L일 trend
    """
    L = len(last_trend)
    x = np.arange(L)
    # 최소제곱 선형 회귀
    coef = np.polyfit(x, last_trend, deg=1)
    a, b = coef[0], coef[1]  # y = a*x + b
    x_future = np.arange(L, L + horizon)
    return a * x_future + b

def repeat_seasonal_weekly(last_seasonal: np.ndarray, horizon: int) -> np.ndarray:
    """
    주기=7 가정. 마지막 7일 seasonal 패턴을 반복.
    last_season = last_seasonal[-7:]
    """
    if len(last_seasonal) >= 7:
        base = last_seasonal[-7:]
    else:
        # 안전장치: 7일보다 짧으면 평균으로 채움
        base = np.full(7, last_seasonal.mean() if len(last_seasonal) > 0 else 0.0)
    reps = int(np.ceil(horizon / 7))
    tiled = np.tile(base, reps)
    return tiled[:horizon]