# 파일 경로: C:\TFT_helper\modules\game_state.py

import cv2
import numpy as np
import pytesseract
import requests
import urllib3

# 🚨 Tesseract 경로 (본인 환경에 맞게 유지)
pytesseract.pytesseract.tesseract_cmd = r'C:\TFT_helper\Tesseract-OCR\tesseract.exe'

# API 경고 끄기
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class StatusManager:
    def __init__(self, sct):
        self.sct = sct
        
        # 💰 골드 OCR 좌표
        self.GOLD_REGION = {
            'x_ratio': 0.497, 'y_ratio': 0.818, 'w_ratio': 0.02, 'h_ratio': 0.025      
        }
        
        # 🧪 경험치 OCR 좌표
        self.XP_REGION = {
            'x_ratio': 0.28, 'y_ratio': 0.81, 'w_ratio': 0.025, 'h_ratio': 0.035    
        }

        # 📊 TFT 레벨별 경험치 표 (Set 13)
        self.XP_TABLE = {
            1: 2, 2: 2, 3: 6, 4: 10, 5: 20, 
            6: 36, 7: 48, 8: 80, 9: 84, 10: 999
        }

    def get_player_info(self):
        """[API] 닉네임, 레벨"""
        try:
            url = "https://127.0.0.1:2999/liveclientdata/activeplayer"
            response = requests.get(url, verify=False, timeout=0.5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'name': data.get('summonerName', 'Unknown'),
                    'level': data.get('level', 0)
                }
        except:
            pass
        return {'name': 'Loading...', 'level': 0}

    def get_gold(self, window_rect):
        """[OCR] 골드 읽기"""
        if not window_rect: return -1
        
        text = self._ocr_process(
            window_rect, self.GOLD_REGION, 
            threshold=215,          
            whitelist="0123456789", 
            thickness=0             
        )
        if text and text.isdigit():
            return int(text)
        return -1

    def get_xp(self, window_rect, current_level):
        """[OCR] 경험치 읽기"""
        if not window_rect: return -1, -1, ""

        max_xp = self.XP_TABLE.get(current_level, 99)

        # 1차 시도
        text = self._ocr_process(
            window_rect, self.XP_REGION, 
            threshold=160, whitelist="0123456789/", 
            thickness=-1  
        )
        curr_xp = self._parse_first_number(text)
        if curr_xp != -1:
            return curr_xp, max_xp, text

        # 2차 시도
        text_retry = self._ocr_process(
            window_rect, self.XP_REGION, 
            threshold=150, whitelist="0123456789/", 
            thickness=0
        )
        curr_xp = self._parse_first_number(text_retry)
        if curr_xp != -1:
            return curr_xp, max_xp, text_retry

        return -1, max_xp, text

    def _parse_first_number(self, text):
        if not text: return -1
        try:
            first_part = str(text).split('/')[0]
            clean_num = first_part.strip()
            if clean_num.isdigit():
                return int(clean_num)
        except:
            pass
        return -1

    def _ocr_process(self, window_rect, region_conf, threshold, whitelist, thickness=0):
        x, y, w, h = window_rect['left'], window_rect['top'], window_rect['width'], window_rect['height']
        gx = int(x + (w * region_conf['x_ratio']))
        gy = int(y + (h * region_conf['y_ratio']))
        gw = int(w * region_conf['w_ratio'])
        gh = int(h * region_conf['h_ratio'])
        
        region = {'top': gy, 'left': gx, 'width': gw, 'height': gh}
        img = np.array(self.sct.grab(region))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) # 바로 GRAY 변환
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(binary)
        
        scaled = cv2.resize(inverted, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        if thickness != 0:
            kernel = np.ones((2, 2), np.uint8)
            if thickness > 0:
                scaled = cv2.erode(scaled, kernel, iterations=thickness)
            else:
                scaled = cv2.dilate(scaled, kernel, iterations=abs(thickness))
        
        final_img = cv2.copyMakeBorder(scaled, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        try:
            return pytesseract.image_to_string(final_img, config=config).strip()
        except:
            return None