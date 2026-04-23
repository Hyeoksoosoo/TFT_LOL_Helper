# modules/shop.py
import cv2
import os
import json

class ShopRecognizer:
    def __init__(self, template_path, champ_json_path, traits_json_path):
        self.templates = {}
        self.champ_data = {}
        self.traits_data = {}
        
        # 설정된 좌표 (사용자 환경 3440 x 1440)
        self.SHOP_W = 254
        self.SHOP_H = 186
        self.SHOP_SLOTS = [
            (1207, 1333), (1475, 1333), (1746, 1332), (2012, 1332), (2282, 1332)
        ]

        # 데이터 로드
        self._load_templates(template_path)
        self._load_json(champ_json_path, 'champ')
        self._load_json(traits_json_path, 'trait')

    def _load_templates(self, path):
        if not os.path.exists(path): return
        for f in os.listdir(path):
            if f.lower().endswith(('.png', '.jpg')):
                name = os.path.splitext(f)[0]
                img = cv2.imread(os.path.join(path, f))
                if img is not None: self.templates[name] = img

    def _load_json(self, path, type):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                if type == 'champ': self.champ_data = json.load(f)
                elif type == 'trait': self.traits_data = json.load(f)

    def recognize(self, full_img):
        """상점 5칸 인식 결과 반환"""
        results = []
        half_w, half_h = self.SHOP_W // 2, self.SHOP_H // 2

        for cx, cy in self.SHOP_SLOTS:
            x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
            x2, y2 = min(full_img.shape[1], cx + half_w), min(full_img.shape[0], cy + half_h)
            slot_img = full_img[y1:y2, x1:x2]

            if slot_img.size == 0:
                results.append("Empty")
                continue

            best_match, max_val = "Empty", 0
            for name, tmpl in self.templates.items():
                # 크기 보정
                if slot_img.shape != tmpl.shape:
                    slot_img = cv2.resize(slot_img, (tmpl.shape[1], tmpl.shape[0]))
                
                res = cv2.matchTemplate(slot_img, tmpl, cv2.TM_CCOEFF_NORMED)
                if res.max() > max_val:
                    max_val = res.max()
                    best_match = name

            results.append(best_match if max_val >= 0.55 else "Empty")
        return results

    def get_display_info(self, champ_key):
        """화면 출력용 문자열 생성"""
        if champ_key == "Empty": return "❌ [빈 슬롯]"
        
        info = self.champ_data.get(champ_key, {})
        name = info.get("name", champ_key)
        cost = info.get("cost", "?")
        traits = info.get("traits", [])
        
        # 시너지 정보 포맷팅
        traits_str = f"[{', '.join(traits)}]" if traits else ""
        return f"🟢 {name}({cost}코) {traits_str}"