# ===== 멀티프로세싱 워커 모듈 =====
# Windows/Linux 모두 multiprocessing이 "spawn" 방식일 수 있어서, 각 워커 프로세스가
# 시작될 때 이 함수를 다시 "import" 할 수 있어야 함 (노트북 셀에 직접 정의하면 안 됨).
# 그래서 워커 함수만 여기 별도 파일로 뺐음.
#
# 각 워커 프로세스는 자기 것 PaddleOCR 인스턴스를 딱 한 번만 로딩해서 재사용함
# (프로세스 전역 변수에 캐시). 설정은 predict.ipynb의 "OCR 모델" 셀과 동일해야 함.
#
# 2단계(1차 빠르게 + 2차 실패한 것만 민감하게) 파이프라인이라, 모델도 두 개를 따로 둠:
# - process_one: 1차, box_thresh=0.7 (전체 이미지, 빠름 - 2400초 예산 안에 안전하게 들어옴)
# - process_one_sensitive: 2차, box_thresh=0.65 (1차에서 완전미인식이었던 것만 재시도)
#
# 채점 서버는 오프라인이라, PaddleOCR 기본 동작(최초 실행 시 인터넷에서 자동 다운로드)이
# 그대로 통하지 않음. download_weights.sh로 미리 받아둔 ./weights 폴더가 있으면 그
# 로컬 경로의 가중치만 사용하며, 가중치가 없으면 자동 다운로드를 시도하지 않고 오류를 발생시킴.

import time
from pathlib import Path

import cv2

_ocr = None             # 1차용 (box_thresh=0.7) - 프로세스별로 따로 로딩됨
_ocr_sensitive = None   # 2차용 (box_thresh=0.65) - 프로세스별로 따로 로딩됨

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_DET_MODEL_NAME = "PP-OCRv5_mobile_det"
_REC_MODEL_NAME = "korean_PP-OCRv5_mobile_rec"
_DET_MODEL_DIR = _WEIGHTS_DIR / _DET_MODEL_NAME
_REC_MODEL_DIR = _WEIGHTS_DIR / _REC_MODEL_NAME


def _local_model_dir_kwargs():
    """사전 다운로드된 로컬 가중치만 사용한다.
    가중치가 없으면 자동 다운로드를 시도하지 않고 즉시 오류를 발생시킨다."""
    if not _DET_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Detection model not found: {_DET_MODEL_DIR}. "
            "Run download_weights.sh before inference."
        )

    if not _REC_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Recognition model not found: {_REC_MODEL_DIR}. "
            "Run download_weights.sh before inference."
        )

    return {
        "text_detection_model_dir": str(_DET_MODEL_DIR),
        "text_recognition_model_dir": str(_REC_MODEL_DIR),
    }


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            lang="korean",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=_DET_MODEL_NAME,
            text_recognition_model_name=_REC_MODEL_NAME,
            enable_mkldnn=False,
            text_det_box_thresh=0.7,
            text_recognition_batch_size=1,
            **_local_model_dir_kwargs(),
        )
    return _ocr


def _get_ocr_sensitive():
    global _ocr_sensitive
    if _ocr_sensitive is None:
        from paddleocr import PaddleOCR
        _ocr_sensitive = PaddleOCR(
            lang="korean",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=_DET_MODEL_NAME,
            text_recognition_model_name=_REC_MODEL_NAME,
            enable_mkldnn=False,
            text_det_box_thresh=0.65,
            text_recognition_batch_size=1,
            **_local_model_dir_kwargs(),
        )
    return _ocr_sensitive


def _resize_long_side(img, max_side=1500):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _run(image_path_str, get_model_fn):
    # image_id는 확장자를 제외한 파일명이어야 함(제출 스펙 명시) -> .stem 사용.
    # (.name을 쓰면 "000001.jpg"처럼 확장자가 남아서 채점 스키마와 안 맞음)
    t0 = time.time()
    image_path = Path(image_path_str)
    image_id = image_path.stem
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return image_id, "", round(time.time() - t0, 3)
        img = _resize_long_side(img)
        ocr = get_model_fn()
        results = list(ocr.predict(img))
        words = [t for res in results for t in res.get("rec_texts", [])]
        text = " ".join(words)
    except Exception as e:
        text = f"[에러] {e}"
    return image_id, text, round(time.time() - t0, 3)


def process_one(image_path_str):
    """1차 워커: box_thresh=0.7. 이미지 1장을 OCR만 처리해서 (image_id, text, seconds) 반환.
    날짜 후보 추출/파싱/스코어링은 여기서 안 하고 메인 프로세스(predict.ipynb)에서 처리함."""
    return _run(image_path_str, _get_ocr)


def process_one_sensitive(image_path_str):
    """2차 워커: box_thresh=0.65. 1차에서 완전미인식(NONE)이었던 이미지만 넘겨서 재시도."""
    return _run(image_path_str, _get_ocr_sensitive)
