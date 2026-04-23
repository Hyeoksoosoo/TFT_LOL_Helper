# config.py
import os

# ==============================
# ⚙️ Tesseract 경로 설정
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_PATH = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")
TESSDATA_PATH = os.path.join(BASE_DIR, "Tesseract-OCR", "tessdata")

# ==============================
# 🖥️ 해상도 및 좌표 설정 (3440 x 1440 기준)
# ==============================
# 골드 위치 비율 (이전에 찾으신 값)
GOLD_REGION = {
    'x_ratio': 0.496,
    'y_ratio': 0.81,
    'w_ratio': 0.02,
    'h_ratio': 0.037    
}

# 2. 아이템 (Items) - 이미지 매칭용
# 사용자가 직접 찾은 황금 좌표 (세로 배치)
ITEM_REGION = {
    'start_x_ratio': 0.004,
    'start_y_ratio': 0.256,
    'box_size': 55,     # 아이콘 크기
    'gap': 68,          # 아이콘 간격
    'direction': 'vertical' # 세로 방향
}
# (나중에 추가될 아이템, 상점 좌표도 여기에 추가 예정)