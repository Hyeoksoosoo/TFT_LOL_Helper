import cv2
import numpy as np
import mss
import os
import sys
import shutil
from ultralytics import YOLO

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.window import find_game_window
from utils.board_mapper import BoardMapper

# =========================================================
# ⚙️ 핵심 설정 (튜닝 포인트)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'tft_model', 'champion_classifier2', 'weights', 'best.pt')

# 1. 캡처 크기 (박스 크기에 맞춤)
CROP_SIZE = 70 

# 2. 시선 보정 (매우 중요!)
# 챔피언 발바닥이 아닌 얼굴/몸통을 보기 위해 시선을 위로 올립니다.
# 인식이 잘 안 되면 이 값을 30~60 사이로 조절하세요.
OFFSET_Y = 40 

# 3. 민감도 설정
# 빈 땅인데 챔피언으로 인식하면 -> 값을 올리세요 (예: 20)
# 챔피언이 있는데 무시하면 -> 값을 내리세요 (예: 10)
EMPTY_THRESHOLD = 15

# 4. AI 확신도 커트라인 (0.4 = 40% 이상 확신할 때만 인정)
CONFIDENCE_CUTOFF = 0.5

class DeckAnalyzer:
    def __init__(self):
        print(f"🧠 AI 모델 로딩 중... ({MODEL_PATH})")
        if not os.path.exists(MODEL_PATH):
            print("❌ 모델 파일이 없습니다! 경로를 확인하세요.")
            sys.exit(1)
            
        self.model = YOLO(MODEL_PATH)
        self.mapper = BoardMapper()
        self.sct = mss.mss()
        
        # 디버깅용 폴더 생성 (AI가 본 사진 저장)
        self.debug_dir = os.path.join(CURRENT_DIR, "debug_crops")
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir)
        print(f"📁 디버그 폴더 준비 완료: {self.debug_dir}")

    def is_occupied(self, image):
        """빈 땅인지 유닛인지 색상 변화량으로 판단"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        return std_dev > EMPTY_THRESHOLD

    def analyze_board(self):
        window_rect = find_game_window()
        if not window_rect: return

        x, y, w, h = window_rect['left'], window_rect['top'], window_rect['width'], window_rect['height']
        monitor = {'top': y, 'left': x, 'width': w, 'height': h}
        
        # 전체 화면 캡처
        full_img = np.array(self.sct.grab(monitor))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)
        
        # 슬롯 리스트 합치기
        all_slots = []
        for s in self.mapper.BOARD_SLOTS: s['type']='Board'; all_slots.append(s)
        for s in self.mapper.BENCH_SLOTS: s['type']='Bench'; all_slots.append(s)

        detected_units = []
        print(f"\n🔍 스캔 시작! (Offset Y: -{OFFSET_Y}px)")

        for slot in all_slots:
            cx, cy = slot['center']
            
            # 🚨 [핵심] 시선 위로 올리기 (발바닥 -> 몸통)
            cy = cy - OFFSET_Y
            
            # 이미지 잘라내기 (Crop)
            half = CROP_SIZE // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)
            
            crop = full_img[y1:y2, x1:x2]
            if crop.size == 0: continue

            # 1. 빈 칸 체크
            if not self.is_occupied(crop):
                # 빈 칸은 회색 박스
                cv2.rectangle(full_img, (x1, y1), (x2, y2), (50, 50, 50), 1)
                continue

            # 2. AI 예측
            results = self.model.predict(crop, verbose=False)
            top1_idx = results[0].probs.top1
            name = results[0].names[top1_idx]
            conf = results[0].probs.top1conf.item()

            # 📸 디버깅용 이미지 저장 (AI가 뭘 봤는지 확인용)
            r, c = slot['index']
            debug_name = f"{slot['type']}_{r}_{c}_pred_{name}_{int(conf*100)}.jpg"
            cv2.imwrite(os.path.join(self.debug_dir, debug_name), crop)

            # 3. 결과 필터링
            if conf > CONFIDENCE_CUTOFF:
                detected_units.append({
                    'champion': name, 
                    'conf': conf, 
                    'pos': slot['index'],
                    'type': slot['type']
                })
                
                # 인식 성공: 빨간 박스 + 이름
                cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(full_img, f"{name}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            else:
                # 확신이 부족함: 노란 박스
                cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # 최종 결과 저장 및 출력
        cv2.imwrite("result_analysis.png", full_img)
        
        print("\n" + "="*40)
        print(f"📊 최종 인식 결과: {len(detected_units)}명 발견")
        print("="*40)
        for unit in detected_units:
            print(f"[{unit['type']}] {unit['pos']} : {unit['champion']} ({unit['conf']*100:.1f}%)")
            
        print("\n✅ 분석 끝!")
        print(f"1. 결과 이미지 확인: {os.path.abspath('result_analysis.png')}")
        print(f"2. AI 시점 확인: {os.path.abspath(self.debug_dir)}")

if __name__ == "__main__":
    DeckAnalyzer().analyze_board()