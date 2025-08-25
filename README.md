# Foodservice Demand Forecast
LG Aimers 해커톤 - 리조트 내 식음업장 메뉴 수요 예측 AI 프로젝트  

## 배경
리조트 내 식음업장은 **계절, 요일, 투숙객 수, 행사 일정** 등 다양한 요인에 따라 수요가 크게 변동합니다.  
특히 휴양지 리조트는 단기간에 집중되는 고객 수요와 예측하기 어려운 방문 패턴으로 인해,  
메뉴별 식자재 준비, 인력 배치, 재고 관리에 있어 높은 운영 난이도를 가집니다.  

정확한 메뉴 수요 예측은 비용 절감과 고객 만족도 향상의 핵심 요소이며,  
최근에는 **AI 기반 수요 예측**이 식음 서비스 운영의 새로운 해법으로 주목받고 있습니다.  

## 주제
**리조트 내 식음업장 메뉴별 1주일 수요 예측 AI 모델 개발**

- 과거의 메뉴별 판매 데이터를 활용하여 향후 1주일간 메뉴별 예상 판매량을 예측
- 데이터 기반 의사결정을 통해 **재고 관리 최적화, 인력 배치 효율화, 고객 경험 개선**에 기여

## 프로젝트 개요
- 시계열 데이터를 **T+S 분해(STL)** 후, 잔차에 대해 Transformer 모델을 적용
- 이벤트성 변동(`is_event`) 및 이상치(`is_outlier`) 플래그를 생성하여 모델에 피처로 활용

### 폴더 구조
```
LG-AIMERS/
│
├─ dataset/
│   ├─ custom_dataset.py      # 잔차 계산 슬라이딩 윈도우 ?
│   └─ split_time.py          # 학습 및 검증 데이터 분리 ?
│
├─ models/
│   └─ transformer_model.py   # Transformer 기반 모델 정의
│
├─ preprocess/
│   ├─ notyet/
│   │   ├─ clustering.py      # 메뉴명 가게 클러스터링
│   │   └─ embedding.py       # 임베딩
│   ├─ calendar.py            # 달력 관련 전처리 ㅇ
│   ├─ encoders.py            # store, menu, holiday 라벨 ㅇ
│   └─ data_loader.py         # 데이터 불러오기 및 전처리 ㅇ
│
├─ stl/
│   ├─ notyet/
│   │   └─ rolling_stats.py   # rolling mean/std
│   ├─ stl_decompose.py       # STL 시계열 분해 ㅇ
│   └─ trend_extrapolate.py   # 미래 트렌드 외삽 ?
│
├─ train.py                   # 학습 진입점
├─ inference.py               # 예측 진입점
│
└─ notebooks/                 # 실험용 Jupyter Notebook (EDA, 테스트)
```

### 환경설정 명령어 모음 (WSL + Python 가상환경)
```bash
# 1. WSL 실행(종료는 exit)
wsl

# 2. 프로젝트 폴더 생성 및 이동
mkdir myproject && cd myproject

# 3. 가상환경(venv) 설치
sudo apt update
sudo apt install python3.10-venv

# 4. 가상환경 생성
python3 -m venv .venv

# 5. 가상환경 활성화(나가는건 deactivate)
source .venv/bin/activate

# 6. pip 업그레이드
python3 -m pip install --upgrade pip

# 7. 필요한 패키지 설치
python3 -m pip install ultralytics
python3 -m pip install notebook
...

# 8. 패키지 버전 저장
pip freeze > requirements.txt

# 9. (다른 환경에서) 패키지 일괄 설치 (venv 활성화 상태)
pip install -r requirements.txt

# 10. VSCode 실행
code . (ctrl+shift+p 눌러서 Python: Select Interpreter 입력 후 venv 선택)

# 11. Jupyter Notebook 실행

jupyter notebook




