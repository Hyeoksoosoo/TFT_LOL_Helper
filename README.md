# TFT LOL Helper

## 프로젝트 개요
TFT LOL Helper는 전략적 팀 전투(Teamfight Tactics) 게임 데이터를 분석하고, 시너지 및 아이템 정보를 추적하며, YOLO 모델을 활용한 이미지 분석 기능을 제공하는 도구입니다. 이 프로젝트는 Python을 기반으로 구축되었으며, 실무형 아키텍처를 적용하여 코드와 데이터를 효율적으로 분리 관리합니다.

---

## 주요 기능
- **YOLO 모델 테스트 및 학습**
  - YOLOv11 모델을 활용한 실시간 유닛 및 아이템 인식
  - `scripts/` 폴더 내 스크립트를 통해 모델 테스트 및 학습 가능

- **데이터셋 생성 및 관리**
  - `data/` 폴더 내에서 챔피언, 시너지, 상점 관련 JSON 데이터 관리
  - 전처리된 학습 데이터셋은 `data/datasets/`에서 안전하게 보관

- **OCR(문자 인식)**
  - Tesseract-OCR을 활용한 텍스트 추출
  - `src/utils/ocr.py`를 통해 게임 내 텍스트 정보 수집

- **게임 상태 추적**
  - `src/modules/` 내의 모듈화된 로직으로 보드 상태, 아이템 보유 현황, 상점 목록 등을 추적

- **분석 및 시각화**
  - `scripts/run_analysis.py`를 통해 수집된 데이터 분석 결과 시각화

---

## 폴더 구조
```text
TFT_helper/
├── data/                   # 📁 데이터 및 설정 파일
│   ├── datasets/           # 학습용 이미지 데이터셋
│   ├── champion_scales.json
│   └── traits.json
├── models/                 # 📁 모델 가중치 및 OCR 엔진
│   ├── tft_model/          # 학습된 YOLO 모델 (.pt)
│   └── Tesseract-OCR/      # OCR 엔진 관련 파일
├── scripts/                # 📁 실행 스크립트 (기능별 분리)
│   ├── train_yolo.py       # 모델 학습 실행
│   ├── test_yolo.py        # 모델 테스트 실행
│   ├── run_analysis.py     # 데이터 분석 실행
│   └── generate_data_label.py
├── src/                    # ⭐️ 핵심 소스 코드 (Core Logic)
│   ├── config.py           # 경로 및 환경 설정
│   ├── modules/            # 게임 로직 (board, items, shop 등)
│   └── utils/              # 유틸리티 (ocr, window 제어 등)
├── main.py                 # 🚀 프로그램 메인 진입점
├── .gitignore              # 대용량 파일 및 가상환경 제외 설정
├── requirements.txt        # 의존성 라이브러리 목록
└── README.md               # 프로젝트 가이드
```

---

## 설치 및 실행 방법

### 1. 의존성 설치
Python 환경(3.10 이상 권장)에서 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

### 2. 데이터 및 모델 준비
- 학습된 모델 파일(`.pt`)은 `models/tft_model/` 폴더에 배치합니다.
- 게임 분석에 필요한 데이터셋은 `data/datasets/`에 위치시킵니다.
- **참고:** 대용량 파일은 보안 및 용량 관리를 위해 Git 추적에서 제외되어 있습니다.

### 3. 스크립트 실행
- **프로그램 전체 실행:** `python main.py`
- **YOLO 모델 테스트:** `python scripts/test_yolo.py`
- **모델 학습 시작:** `python scripts/train_yolo.py`
- **데이터 분석 실행:** `python scripts/run_analysis.py`

---

## 주의사항
- **환경 분리:** 맥북(MacBook)과 윈도우 환경 모두 지원하도록 경로 설정이 `src/config.py`에 최적화되어 있습니다.
- **용량 관리:** 데이터셋과 학습 결과물이 포함된 `data/`, `TFT_Project/`, `tft_model/` 등은 Git에 포함되지 않으므로 별도 백업이 필요합니다.
- **OCR 설정:** Tesseract-OCR이 시스템에 설치되어 있어야 하며, 상세 경로는 `src/config.py`에서 확인 가능합니다.

---

## 기여 방법
1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다: `git checkout -b feature/AmazingFeature`
3. 변경 사항을 커밋합니다: `git commit -m 'Add some AmazingFeature'`
4. 브랜치에 푸시합니다: `git push origin feature/AmazingFeature`
5. Pull Request를 생성합니다.

---

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.
