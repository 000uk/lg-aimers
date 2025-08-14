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

3. 이벤트 플래그에서의 누수
예: add_event(df, window=28)

만약 이게 “앞뒤 28일 안에 이벤트가 있는지”를 보고 만든 플래그라면,
t 시점에서 앞으로 10일 뒤에 있을 이벤트도 알게 됨 → 누수.

해결

예측 시점에는 미래 이벤트 정보가 이미 공개된 경우만 허용
(예: 공휴일, 예정된 프로모션)

그렇지 않으면 **왼쪽 창(과거 데이터만)**을 써서 이벤트 파생.

4. 이상치 플래그에서의 누수
예: rolling mean ± k×std로 이상치 판단

중앙 기준(centered) 윈도우 쓰면 t 시점 이상치 판정에 미래 값 포함됨 → 누수.

해결

이상치 탐지할 때도 left-window(과거 데이터만) 사용.

5. 정리
train만 적합: 피처 생성 시점에서 미래 관측치가 포함되지 않게, train 데이터만 보고 fit

예측구간은 반복/외삽/left-window:

seasonal → 주기 반복

trend → 과거 구간 외삽

이벤트/이상치 → 과거 정보만 참조

원하면 내가 STL + 이벤트 + 이상치 플래그를
전부 누수 없이 만드는 코드 샘플로 묶어서 줄 수 있습니다.
그러면 train/inference 파이프라인에서 그대로 쓸 수 있어요.
"""

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