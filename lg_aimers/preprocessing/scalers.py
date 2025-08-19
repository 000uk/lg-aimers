"""
그룹별 normalization은,
store나 menu 같이 여러 시계열이 섞여 있는 데이터에서,
같은 그룹(시리즈) 안에서만 평균과 표준편차를 계산해 정규화하는 걸 말해요.

왜 쓰나?
멀티시계열에서 전체 데이터를 한 번에 z-score 하면,
판매량이 큰 매장/메뉴가 작은 매장/메뉴의 패턴을 덮어버릴 수 있어요.

예:

A 매장: 하루 평균 500개 판매

B 매장: 하루 평균 5개 판매
전체 평균으로 스케일링하면 B 매장은 거의 0에 가까운 값만 나옴 → 모델이 학습하기 힘듦.

그래서 **그룹별(매장·메뉴별)**로 평균/표준편차를 따로 계산해서 스케일링합니다.
"""
import pandas as pd

def sales_scaler(df):
  df = df.copy()

  # transform: 원본 DataFrame 크기와 맞춰 각 행에 정규화 값 할당
  df['sales_norm'] = df.groupby('store_menu')['sales_qty'].transform(
    lambda x: (x - x.mean()) / x.std()
  ).fillna(0) # x.std()가 0이면 NaN이 나옴
  
  return df