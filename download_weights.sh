#!/usr/bin/env bash
# ITDA3 CODE-MIN - PaddleOCR 가중치 사전 다운로드 스크립트
#
# 채점 서버는 오프라인이라 predict.ipynb Run All 도중에는 인터넷에서 모델을
# 새로 받을 수 없습니다. 이 스크립트를 인터넷이 되는 환경에서 "채점 실행 전에
# 딱 1회" 실행해서, 필요한 PaddleOCR 모델 2개를 이 저장소의 ./weights 폴더에
# 내려받아 두세요. predict.ipynb / ocr_worker.py는 ./weights가 존재하면 그
# 로컬 경로에서만 모델을 로딩하고, 존재하지 않으면 그때 자동 다운로드를
# 시도합니다(오프라인 환경에서는 이 경우 에러가 납니다).
#
# 사용법:
#   pip install -r requirements.txt
#   bash download_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

echo "[download_weights] PaddleOCR 모델(PP-OCRv5_mobile_det, korean_PP-OCRv5_mobile_rec)을"
echo "[download_weights] 기본 캐시로 내려받은 뒤 ./weights 로 복사합니다..."

python3 - "$WEIGHTS_DIR" <<'PYEOF'
import shutil
import sys
from pathlib import Path

weights_dir = Path(sys.argv[1])

DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "korean_PP-OCRv5_mobile_rec"

# predict.ipynb/ocr_worker.py와 동일한 설정으로 인스턴스를 한 번 만들면, 아직 로컬에
# 없는 모델만 PaddleX가 자동으로 기본 캐시(~/.paddlex/official_models)에 내려받는다.
# use_doc_orientation_classify 등을 전부 꺼둔 것도 동일하게 맞춰서, 우리가 실제로
# 쓰지 않는 부가 모델(방향 보정 등)까지 불필요하게 다운로드되지 않도록 한다.
from paddleocr import PaddleOCR

PaddleOCR(
    lang="korean",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name=DET_MODEL,
    text_recognition_model_name=REC_MODEL,
    enable_mkldnn=False,
    text_det_box_thresh=0.7,
    text_recognition_batch_size=1,
)

cache_root = Path.home() / ".paddlex" / "official_models"

for model_name in (DET_MODEL, REC_MODEL):
    src = cache_root / model_name
    dst = weights_dir / model_name
    if not src.exists():
        raise SystemExit(f"[download_weights] 모델 캐시를 찾을 수 없음: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[download_weights] {model_name} -> {dst}")

print("[download_weights] 완료. predict.ipynb Run All은 이제 이 폴더의 가중치만 사용합니다.")
PYEOF

echo "[download_weights] weights 폴더 내용:"
ls -la "$WEIGHTS_DIR"
