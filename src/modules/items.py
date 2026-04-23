import cv2
import numpy as np
import os
from config import ITEM_REGION

class ItemManager:
    def __init__(self, sct):
        self.sct = sct
        self.templates = {} 
        self.load_templates()

    def load_templates(self):
        """
        도감 이미지를 원본 크기 그대로 로딩합니다.
        """
        path = "item_images/known"
        if not os.path.exists(path):
            return

        count = 0
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(path, filename)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[name] = img
                    count += 1
        
        print(f"📦 [아이템 로더] {count}개 아이템 도감 로딩 완료 (원본 크기 유지)")

    def capture_slots(self, window_rect):
        """
        화면 캡처 (설정된 box_size 만큼)
        """
        if not window_rect: return []

        r = ITEM_REGION
        box_size = int(r['box_size']) 

        start_x = int(window_rect['left'] + (window_rect['width'] * r['start_x_ratio']))
        start_y = int(window_rect['top'] + (window_rect['height'] * r['start_y_ratio']))
        
        captured_images = []

        for i in range(10):
            curr_x = start_x
            curr_y = start_y + (i * r['gap'])
            
            region = {
                "top": int(curr_y),
                "left": int(curr_x),
                "width": box_size,
                "height": box_size
            }
            
            try:
                img = np.array(self.sct.grab(region))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                captured_images.append(img)
            except Exception:
                pass
            
        return captured_images

    def identify_item(self, target_img):
        """
        [숨은 그림 찾기] 큰 타겟 이미지 안에서 작은 템플릿을 찾습니다.
        """
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # 1. 빈 칸 체크 (밝기 기준)
        if np.mean(gray_target) < 30: 
            return None

        best_match_name = None
        best_match_score = 0.0

        t_h, t_w = gray_target.shape[:2]

        # 2. 도감과 비교
        for name, template in self.templates.items():
            temp_h, temp_w = template.shape[:2]

            # 상황 A: 도감(42px)이 캡처(55px)보다 작을 때 (정상) -> "그냥 찾기"
            if temp_h <= t_h and temp_w <= t_w:
                res = cv2.matchTemplate(gray_target, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(res)
            
            # 상황 B: 도감 이미지가 캡처보다 클 때 -> "도감을 줄여서 찾기"
            else:
                # 캡처 크기에 딱 맞게 축소
                resized_temp = cv2.resize(template, (t_w, t_h))
                res = cv2.matchTemplate(gray_target, resized_temp, cv2.TM_CCOEFF_NORMED)
                score = np.max(res)

            # 상황 C: 스케일링 매칭 (크기가 미묘하게 안 맞을 때를 대비한 필살기)
            # 도감 이미지를 90% ~ 110% 크기로 조절해보며 매칭
            if score < 0.8: # 점수가 애매할 때만 정밀 검사
                for scale in [0.9, 1.0, 1.1]:
                    new_w = int(temp_w * scale)
                    new_h = int(temp_h * scale)
                    
                    # 캡처보다 커지면 패스
                    if new_w > t_w or new_h > t_h: continue
                    
                    resized_temp = cv2.resize(template, (new_w, new_h))
                    res = cv2.matchTemplate(gray_target, resized_temp, cv2.TM_CCOEFF_NORMED)
                    new_score = np.max(res)
                    
                    if new_score > score:
                        score = new_score

            # 최고 점수 갱신
            if score > best_match_score:
                best_match_score = score
                best_match_name = name

        # 3. 정확도 판별 (기준: 0.7)
        if best_match_score > 0.7:
            return best_match_name
        else:
            return None # 점수가 낮으면 인식 실패 처리