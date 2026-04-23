import cv2
import numpy as np
import mss
import os
import glob
import math
import json
import time
import pytesseract

# ==========================================
# ⚙️ 1. 설정 및 경로
# ==========================================
# ★ Tesseract 설치 경로 (본인 PC에 맞게 수정 필수)
pytesseract.pytesseract.tesseract_cmd = r'C:\TFT_helper\Tesseract-OCR\tesseract.exe'

BASE_DIR = r"C:\TFT_helper"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
ICON_DIR = os.path.join(DATA_DIR, "synergy_raw") 
TRAITS_FILE = os.path.join(DATA_DIR, "traits.json")

# 이미지 처리 설정
LIVE_GAMMA = 4.0
LIVE_BLUR = 3
LIVE_MIN = 0
DATA_GAMMA = 3.0
DATA_BLUR = 1
DATA_MIN = 0

MASK_RADIUS = 28
SEARCH_RADIUS = 45
MATCH_THRESHOLD = 0.60

# ★ 확정된 아이콘 좌표
FIXED_POINTS = [
    (109, 410), # Slot 1
    (109, 477), # Slot 2
    (109, 544), # Slot 3
    (109, 611), # Slot 4
    (109, 679), # Slot 5
    (109, 746), # Slot 6
    (109, 813), # Slot 7
    (109, 881), # Slot 8
    (109, 947), # Slot 9
]

# ★ 텍스트(인원수) 영역 설정값
TEXT_ROIS = [
    {'dx': 50, 'dy': 1, 'w': 34, 'h': 45}, # Slot 1
    {'dx': 50, 'dy': 2, 'w': 34, 'h': 45}, # Slot 2
    {'dx': 49, 'dy': 2, 'w': 34, 'h': 45}, # Slot 3
    {'dx': 49, 'dy': 1, 'w': 34, 'h': 45}, # Slot 4
    {'dx': 49, 'dy': 1, 'w': 34, 'h': 45}, # Slot 5
    {'dx': 49, 'dy': 2, 'w': 34, 'h': 45}, # Slot 6
    {'dx': 49, 'dy': 2, 'w': 34, 'h': 45}, # Slot 7
    {'dx': 49, 'dy': 0, 'w': 34, 'h': 45}, # Slot 8
    {'dx': 49, 'dy': 1, 'w': 34, 'h': 45}, # Slot 9
]

# ==========================================
# 🔧 2. OCR 및 이미지 처리 함수
# ==========================================
def get_ocr_result(img_roi):
    """
    이미지 영역에서 숫자와 슬래시(/)를 읽어옵니다.
    """
    if img_roi.size == 0: return ""
    
    # 1. 전처리 (확대 -> 그레이스케일 -> 이진화)
    # 텍스트 인식을 위해 3배 확대
    scale = 3
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 배경이 어둡고 글씨가 밝으므로 Binary Inversion 사용
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 2. OCR 실행
    # psm 7: 한 줄 텍스트 모드
    # whitelist: 숫자와 / 만 허용
    config = "--psm 7 -c tessedit_char_whitelist=0123456789/"
    try:
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()
    except:
        return ""

def get_tier_info(roi_bgr):
    h, w = roi_bgr.shape[:2]
    center_roi = roi_bgr[h//4:h*3//4, w//4:w*3//4]
    
    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    debug_info = f"(H:{int(mean_h)} S:{int(mean_s)} V:{int(mean_v)})"

    if mean_v < 60: return "흑백(Inactive)", (100, 100, 100), debug_info
    if mean_s < 80: return "실버(Silver)", (192, 192, 192), debug_info
    if (21 <= mean_h <= 45) and mean_v > 100 and mean_s > 80: return "골드(Gold)", (0, 215, 255), debug_info
    if (0 <= mean_h <= 50) or (160 <= mean_h <= 180): return "브론즈(Bronze)", (42, 42, 165), debug_info
    return "프리즘(Prismatic)", (255, 0, 255), debug_info

def apply_gamma(img, gamma=1.0):
    if gamma <= 0: gamma = 0.1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_hex_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    vertices = []
    for i in range(6):
        angle_deg = 30 + (60 * i)
        angle_rad = math.radians(angle_deg)
        x = cx + int(MASK_RADIUS * math.cos(angle_rad))
        y = cy + int(MASK_RADIUS * math.sin(angle_rad))
        vertices.append([x, y])
    pts = np.array(vertices, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(img, img, mask=mask)

def process_live_unified(img):
    boosted = apply_gamma(img, LIVE_GAMMA)
    if len(boosted.shape) == 3: gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    else: gray = boosted
    masked_gray = apply_hex_mask(gray)
    k = int(LIVE_BLUR)
    if k % 2 == 0: k += 1
    blurred = cv2.GaussianBlur(masked_gray, (k, k), 0)
    return cv2.Canny(blurred, LIVE_MIN, 255)

def process_template_exact(img):
    boosted = apply_gamma(img, DATA_GAMMA)
    if len(boosted.shape) == 3: gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    else: gray = boosted
    masked_gray = apply_hex_mask(gray)
    k = int(DATA_BLUR)
    if k % 2 == 0: k += 1
    blurred = cv2.GaussianBlur(masked_gray, (k, k), 0)
    return cv2.Canny(blurred, DATA_MIN, 255)

# ==========================================
# 📂 데이터 로드
# ==========================================
def load_game_data():
    if not os.path.exists(TRAITS_FILE): return {}
    with open(TRAITS_FILE, 'r', encoding='utf-8') as f: data = json.load(f)
    mapping = {}
    EXCEPTIONS = ["targon"] 
    for kr, v in data.items():
        if "name_en" in v:
            key = v["name_en"].lower().replace(" ", "")
            is_unique = False
            if "sets" in v and len(v["sets"]) == 1:
                if v["sets"][0].get("min") == 1:
                    if key not in EXCEPTIONS: is_unique = True
            mapping[key] = { "kr": kr, "en": v["name_en"], "is_unique": is_unique }
    return mapping

def load_all_templates():
    templates = {}
    if not os.path.exists(ICON_DIR): return {}
    files = glob.glob(os.path.join(ICON_DIR, "*.png"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if "_" in name: name = name.rsplit("_", 1)[0]
        key = name.lower().replace(" ", "")
        img = cv2.imread(f)
        if img is not None:
            templates[key] = process_template_exact(img)
    return templates

# ==========================================
# 🚀 메인 실행
# ==========================================
def run_tracker_with_ocr():
    trait_map = load_game_data()
    templates = load_all_templates()
    
    if not templates: 
        print("❌ 템플릿 없음.")
        return

    # Tesseract 경로 확인
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("❌ Tesseract 실행 파일을 찾을 수 없습니다.")
        print(f"경로: {pytesseract.pytesseract.tesseract_cmd}")
        return

    sct = mss.mss()
    monitor = sct.monitors[1]

    cv2.namedWindow("Synergy Tracker", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug: Raw vs Processed", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Debug: Raw vs Processed", 400, 900)

    print("🚀 TFT 시너지 & 인원수(OCR) 트래커 시작")
    last_print_time = 0
    PRINT_INTERVAL = 0.2

    while True:
        full_img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)
        
        detected_results = [] 
        debug_slots = [] 

        for i, (cx, cy) in enumerate(FIXED_POINTS):
            x1, y1 = max(0, cx - SEARCH_RADIUS), max(0, cy - SEARCH_RADIUS)
            x2, y2 = min(frame.shape[1], cx + SEARCH_RADIUS), min(frame.shape[0], cy + SEARCH_RADIUS)
            slot_roi = frame[y1:y2, x1:x2]
            
            if slot_roi.size == 0: 
                debug_slots.append(np.zeros((90, 180, 3), dtype=np.uint8))
                continue

            # 1. 이미지 처리 및 매칭
            processed_slot = process_live_unified(slot_roi)
            best_score = 0.0
            best_key = None

            for key, templ_img in templates.items():
                res = cv2.matchTemplate(processed_slot, templ_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_score:
                    best_score = max_val
                    best_key = key
                if max_val > 0.90: break 

            # 2. 디버그 뷰 생성
            processed_bgr = cv2.cvtColor(processed_slot, cv2.COLOR_GRAY2BGR)
            roi_display = slot_roi.copy()
            # 육각형 가이드
            h_roi, w_roi = roi_display.shape[:2]
            cx_r, cy_r = w_roi // 2, h_roi // 2
            vertices = []
            for k in range(6):
                angle_deg = 30 + (60 * k)
                angle_rad = math.radians(angle_deg)
                vx = cx_r + int(MASK_RADIUS * math.cos(angle_rad))
                vy = cy_r + int(MASK_RADIUS * math.sin(angle_rad))
                vertices.append([vx, vy])
            pts = np.array(vertices, np.int32)
            cv2.polylines(roi_display, [pts], True, (0, 0, 255), 1)

            combined_view = np.hstack((roi_display, processed_bgr))
            cv2.putText(combined_view, f"Slot {i+1}", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_slots.append(combined_view)

            # 3. 결과 판독 (활성 시너지만)
            if best_score >= MATCH_THRESHOLD and best_key:
                info = trait_map.get(best_key, {"kr": best_key, "en": best_key, "is_unique": False})
                
                # 티어 판별
                if info["is_unique"]:
                    tier_name = "고유(Unique)"
                    color_bgr = (36, 105, 234)
                    debug_txt = "JSON-Forced"
                else:
                    tier_name, color_bgr, debug_txt = get_tier_info(slot_roi) 
                
                # ★ 활성 상태일 때만 OCR 실행
                if "Inactive" not in tier_name:
                    # 해당 슬롯의 텍스트 영역 좌표 계산
                    roi_params = TEXT_ROIS[i]
                    # 아이콘 중심(cx, cy) 기준 상대좌표 적용
                    tx = cx + roi_params['dx']
                    ty = cy + roi_params['dy']
                    tw, th = roi_params['w'], roi_params['h']
                    
                    # 텍스트 영역 잘라내기
                    t_x1, t_y1 = max(0, tx - tw // 2), max(0, ty - th // 2)
                    t_x2, t_y2 = min(frame.shape[1], tx + tw // 2), min(frame.shape[0], ty + th // 2)
                    
                    text_roi_img = frame[t_y1:t_y2, t_x1:t_x2]
                    
                    # OCR 실행
                    count_text = get_ocr_result(text_roi_img)
                    if not count_text: count_text = "?" # 인식 실패 시

                    # 결과 저장
                    detected_results.append({
                        "slot": i + 1,
                        "name": info['kr'],
                        "tier": tier_name,
                        "count": count_text, # 인원수
                        "debug": debug_txt,
                        "score": best_score
                    })
                    
                    # 화면 표시
                    # 1) 아이콘 박스
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                    # 2) 텍스트 영역 박스 (노란색)
                    cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), (0, 255, 255), 1)
                    
                    # 라벨 표시
                    label = f"{info['en']} [{count_text}]"
                    cv2.rectangle(frame, (x1, y1 - 15), (x1 + 120, y1), (0,0,0), -1)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

        # 4. 디버그 패널 출력
        if debug_slots:
            debug_panel = np.vstack(debug_slots)
            cv2.imshow("Debug: Raw vs Processed", debug_panel)

        # 메인 창 출력
        cv2.imshow("Synergy Tracker", cv2.resize(frame, None, fx=0.5, fy=0.5))

        # 5. 터미널 출력
        if time.time() - last_print_time > PRINT_INTERVAL:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "="*80)
            print("       📊 TFT 시너지 현황 (인원수 포함)")
            print("="*80)
            
            if not detected_results:
                print("\n   ⚠️  활성화된 시너지가 없습니다.\n")
            else:
                detected_results.sort(key=lambda x: x['slot'])
                for item in detected_results:
                    # 출력 포맷: 슬롯 | 이름 | 티어 | [인원수]
                    print(f" Slot {item['slot']} | {item['name']:<8} | {item['tier']:<14} | Count: [{item['count']}]")
                    
            print("-" * 80)
            print(" [Q] 종료")
            print("="*80)
            last_print_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker_with_ocr()