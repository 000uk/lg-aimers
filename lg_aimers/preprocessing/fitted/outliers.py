"""
시계열 예측에서 "미래 정보 누수(leakage)"를 피하는 방법 얘기예요.
STL, 이벤트 플래그, 이상치 플래그 같은 피처 생성 단계에서 미래 데이터를 참조하면 안 된다는 뜻입니다.

1. 왜 미래 정보 누수가 생기나?
시계열은 과거 → 미래 예측이 목표인데,
전처리 과정에서 미래 시점의 데이터가 은근슬쩍 들어오면,
모델이 “실제로는 모를 정보”를 보고 예측하게 됩니다.
→ 검증 점수는 비현실적으로 높아지고, 실제 배포하면 성능 폭락.

2. STL 분해에서의 누수
STL은 기본적으로 양방향 Loess 스무딩을 써서,
t 시점의 trend/seasonal을 계산할 때 앞뒤 데이터를 같이 씁니다.

그런데 테스트 구간까지 같이 넣고 STL을 돌리면,
예측할 구간의 데이터가 trend/seasonal 계산에 반영됨 → 누수.

해결

STL을 train 구간까지만 돌리고,
예측 구간의 seasonal은 주기 반복으로 전개,
trend는 최근 구간 외삽으로 확장.

또는 causal STL(미래 안 보는 방식)로 구현.

이상치 플래그에서의 누수
예: rolling mean ± k×std로 이상치 판단
중앙 기준(centered) 윈도우 쓰면 t 시점 이상치 판정에 미래 값 포함됨 → 누수.
해결: 이상치 탐지할 때도 left-window(과거 데이터만) 사용.
"""

# 해야함
def add_outlier_flag(
    df, 
    target_col='sales_qty', 
    window=7, 
    outlier_std=3
):
    # ===== Rolling Features =====
    df[f'{target_col}_roll_mean_{window}'] = (
        df[target_col].rolling(window, min_periods=1).mean()
    )
    df[f'{target_col}_roll_std_{window}'] = (
        df[target_col].rolling(window, min_periods=1).std()
    )

    # ===== Outlier Detection (Z-score 방식) =====
    mean = df[target_col].mean()
    std = df[target_col].std()
    z_score = (df[target_col] - mean) / std

    # 이상치 True/False
    df[f'{target_col}_is_outlier'] = abs(z_score) > outlier_std

    return df