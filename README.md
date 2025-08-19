## LG-AIMERS 시계열 예측 프로젝트

### 프로젝트 개요
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

