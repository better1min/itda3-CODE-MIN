## 1. 개요

상품 뒷면 이미지에서 소비기한(유통기한/사용기한) 날짜를 OCR로 추출하는 파이프라인입니다.

- 입력: 상품 뒷면 이미지 폴더 (`ITDA_INPUT_DIR`)
- 출력: 결과 CSV 파일 (`ITDA_OUTPUT_PATH`, 기본값 `./submission.csv`)

## 2. 파일 구조

itda3-CODE-MIN/
├── predict.ipynb          # 메인 추론 노트북
├── ocr_worker.py          # 멀티프로세싱 OCR 워커 모듈
├── requirements.txt       # 실행 환경 패키지 목록
├── README.md              # 실행 가이드
├── .gitignore             # 데이터·가중치·결과 파일 제외 설정
├── download_weights.sh    # PaddleOCR 가중치 다운로드 스크립트
└── weights/               # 다운로드된 모델 가중치 저장 폴더


## 3. 오프라인 채점 환경 대응

패키지 설치:

    pip install -r requirements.txt

모델 가중치 준비:

    bash download_weights.sh

실행:

    export ITDA_INPUT_DIR=./val_images
    export ITDA_OUTPUT_PATH=./submission.csv

    jupyter nbconvert --to notebook --execute predict.ipynb \
        --ExecutePreprocessor.timeout=2400 \
        --output /tmp/executed.ipynb

채점 환경은 인터넷이 차단된 Standard 4-Core vCPU 환경입니다.

- `download_weights.sh`를 통해 필요한 PaddleOCR 가중치를 사전에 `./weights/`에 준비합니다.
- `predict.ipynb`와 `ocr_worker.py`는 로컬 `./weights/` 경로의 모델을 사용합니다.
- 입력 및 출력 경로는 `ITDA_INPUT_DIR`, `ITDA_OUTPUT_PATH` 환경변수로 지정합니다.

## 4. 파이프라인 요약

1차 OCR(box_thresh=0.7)로 전체 이미지를 처리한 뒤 날짜 후보를 추출·파싱·점수화하여 최종 소비기한을 선택합니다.

1차에서 완전히 인식되지 않은 이미지에 한해, 시간 예산을 확인한 뒤 box_thresh=0.65로 2차 OCR을 수행합니다.

최종 결과는 `image_id, year, month, day, final_date` 형식으로 저장합니다.
