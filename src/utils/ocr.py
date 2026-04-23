import cv2
import numpy as np
import pytesseract
import re
import os
from config import TESSERACT_PATH, TESSDATA_PATH

# =========================================================
# ⚙️ Tesseract 엔진 초기화
# =========================================================
# config.py에서 설정한 경로를 가져와서 적용합니다.
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
else:
    print(f"❌ [오류] Tesseract 실행 파일이 없습니다.")
    print(f"    경로 확인: {TESSERACT_PATH}")

def process_and_read_number(image):
    """
    이미지를 받아서 숫자로 읽어주는 핵심 함수
    (3배 확대 + Otsu 이진화 + 여백 추가 + 3단 인식 시도)
    """
    try:
        # 1. 흑백 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        # 2. 3배 확대 (작은 숫자 인식률 대폭 상승)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 3. 이진화 (배경/글자 분리)
        # Otsu 알고리즘: 자동으로 최적의 명암 기준을 찾습니다.
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. 여백 추가 (Padding)
        # 글자가 이미지 끝에 붙으면 인식이 안 되므로, 흰색 테두리를 둘러줍니다.
        padding = 20
        thresh = cv2.copyMakeBorder(thresh, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
        
        # (디버깅용 이미지 저장은 이제 경로 문제 없으니 안심하고 쓰셔도 됩니다)
        # cv2.imwrite("debug_ocr_clean.png", thresh)

        # 5. 인식 시도 (가장 확률 높은 순서대로)
        
        # [시도 1] PSM 7: 한 줄로 인식 (가장 기본)
        config_7 = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config_7)
        numbers = re.findall(r'\d+', text)
        if numbers: return int(numbers[0])

        # [시도 2] PSM 6: 텍스트 블록으로 인식
        config_6 = '--psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config_6)
        numbers = re.findall(r'\d+', text)
        if numbers: return int(numbers[0])
        
        # [시도 3] PSM 13: 규격 외 인식 (Raw Line)
        config_13 = '--psm 13 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config_13)
        numbers = re.findall(r'\d+', text)
        if numbers: return int(numbers[0])

        return -1 # 인식 실패

    except Exception as e:
        print(f"⚠️ OCR 에러 발생: {e}")
        return -1