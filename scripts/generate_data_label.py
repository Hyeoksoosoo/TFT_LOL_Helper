import cv2
import os
import random
import glob
import json
import numpy as np

def generate_dataset_final():
    # ==========================================
    # ⚙️ [최종 설정]
    # ==========================================
    TOTAL_IMAGES = 3000      # 3000장이면 충분히 강력합니다
    VAL_SPLIT = 0.2          # 20%는 검증용 (600장)
    
    # 한 화면에 등장할 유닛 수 (랜덤)
    MIN_UNITS = 3
    MAX_UNITS = 12
    
    # 기준 높이 (튜너에서 맞춘 비율의 기준점)
    BASE_HEIGHT = 200 
    
    # 경로 설정
    CHAMP_DIR = r"C:\generate_data"
    BG_DIR = r"C:\TFT_helper\datasets\backgrounds"
    
    # ★ 가장 중요한 파일: 사용자가 V5 튜너로 만든 설정 파일
    SCALE_FILE = r"C:\TFT_helper\image_scales_v5.json"
    
    # 최종 데이터셋이 저장될 폴더
    OUTPUT_DIR = r"C:\TFT_helper\yolo_dataset_v4"

    # ==========================================
    # 🛠️ 핵심 함수들 (여백제거, 그림자, 증강)
    # ==========================================
    def trim_transparent_borders(img):
        """이미지 여백 자동 제거"""
        if img is None: return None
        if img.shape[2] != 4: return img
        alpha = img[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is None: return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    def add_shadow(bg, x, y, w, h):
        """발 밑 그림자 생성"""
        shadow_w = int(w * 0.8)
        shadow_h = int(h * 0.25)
        cx = x + w // 2
        cy = y + h - int(h * 0.05)
        overlay = bg.copy()
        cv2.ellipse(overlay, (cx, cy), (shadow_w//2, shadow_h//2), 0, 0, 360, (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, bg, 0.6, 0, bg)

    def apply_augmentations(img):
        """데이터 증강 (Sim2Real Gap 줄이기)"""
        aug = img.copy()
        # 1. 밝기/대비 변화 (조명 대응)
        if random.random() < 0.5:
            aug = cv2.convertScaleAbs(aug, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        # 2. 블러 (움직임/해상도 대응)
        if random.random() < 0.2:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)
        return aug

    # ==========================================
    # 📍 메인 로직 준비
    # ==========================================
    class BoardMapper:
        def __init__(self):
            # 3440 해상도 좌표
            self.ALL_SLOTS = [
                {'center': (1189, 592)}, {'center': (1341, 597)}, {'center': (1496, 596)}, {'center': (1649, 599)}, {'center': (1805, 597)}, {'center': (1954, 596)}, {'center': (2107, 594)},
                {'center': (1251, 688)}, {'center': (1411, 690)}, {'center': (1570, 685)}, {'center': (1726, 687)}, {'center': (1885, 690)}, {'center': (2043, 689)}, {'center': (2200, 690)},
                {'center': (1157, 792)}, {'center': (1315, 790)}, {'center': (1485, 793)}, {'center': (1647, 790)}, {'center': (1808, 794)}, {'center': (1970, 790)}, {'center': (2132, 792)},
                {'center': (1216, 897)}, {'center': (1385, 898)}, {'center': (1559, 900)}, {'center': (1726, 896)}, {'center': (1897, 902)}, {'center': (2063, 895)}, {'center': (2234, 902)},
            ]
            for i in range(9): self.ALL_SLOTS.append({'center': (1000 + i * 159, 1050)})
    mapper = BoardMapper()

    # 폴더 생성
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    # 1. 설정 파일 로드
    if os.path.exists(SCALE_FILE):
        with open(SCALE_FILE, 'r') as f: scales = json.load(f)
        print(f"📂 V5 설정 파일 로드 완료 ({len(scales)}개 설정)")
    else: 
        print("⚠️ 설정 파일이 없습니다! 기본값으로 진행합니다. (비추천)")
        scales = {}

    # 2. 챔피언 목록 및 이미지 캐싱
    champ_names = sorted([d for d in os.listdir(CHAMP_DIR) if os.path.isdir(os.path.join(CHAMP_DIR, d))])
    class_map = {name: i for i, name in enumerate(champ_names)}
    
    # data.yaml 자동 생성
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/val\nnc: {len(champ_names)}\nnames: {champ_names}")

    champ_images = {} 
    print("⏳ 이미지 로딩 및 여백 제거(Auto-Crop) 수행 중...")
    
    for name in champ_names:
        imgs = glob.glob(os.path.join(CHAMP_DIR, name, "*.png"))
        loaded_list = []
        for p in imgs:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                trimmed = trim_transparent_borders(img)
                if trimmed is not None:
                    # 파일명 키 생성 (튜너와 동일 규칙: 'Ahri/01.png')
                    key = f"{name}/{os.path.basename(p)}"
                    loaded_list.append({'img': trimmed, 'key': key})
        
        if loaded_list:
            champ_images[name] = loaded_list

    # 3. 배경 로드
    bg_paths = glob.glob(os.path.join(BG_DIR, "*.png")) + glob.glob(os.path.join(BG_DIR, "*.jpg"))
    if not bg_paths: print("❌ 배경 없음"); return

    # ==========================================
    # 🚀 데이터 생성 루프 시작
    # ==========================================
    print(f"\n🚀 총 {TOTAL_IMAGES}장 생성 시작! (잠시만 기다려주세요)")
    
    for i in range(TOTAL_IMAGES):
        if (i+1) % 100 == 0: print(f"   ▶ {i+1} / {TOTAL_IMAGES} 완료...")

        split = 'val' if random.random() < VAL_SPLIT else 'train'
        bg = cv2.imread(random.choice(bg_paths))
        bg_h, bg_w = bg.shape[:2]
        labels = []

        # 랜덤 유닛 배치
        num_units = random.randint(MIN_UNITS, MAX_UNITS)
        selected_slots = random.sample(mapper.ALL_SLOTS, num_units)
        # 중요: Y좌표 기준 정렬 (아래쪽 유닛이 위쪽 유닛을 덮어야 자연스러움)
        selected_slots.sort(key=lambda s: s['center'][1])

        for slot in selected_slots:
            c_name = random.choice(list(champ_images.keys()))
            img_data = random.choice(champ_images[c_name])
            
            # 증강 적용
            c_img = apply_augmentations(img_data['img'])
            img_key = img_data['key']

            # ★ 개별 이미지 설정값 적용
            settings = scales.get(img_key, {"scale": 1.0, "x": 0.0, "y": 0.15})
            
            s_val = settings['scale']
            a_x = settings['x']
            a_y = settings['y']

            # 크기 변환
            target_height = int(BASE_HEIGHT * s_val)
            if target_height < 1: target_height = 1
            
            h_orig, w_orig = c_img.shape[:2]
            scale = target_height / h_orig
            fg_resized = cv2.resize(c_img, None, fx=scale, fy=scale)
            h, w = fg_resized.shape[:2]
            
            cx, cy = slot['center']
            
            # 위치 보정 (앵커 포인트)
            offset_x = int(w * a_x)
            offset_y = int(h * a_y)
            x = int(cx - (w // 2) + offset_x)
            y = int(cy - h + offset_y)

            # 합성 (화면 밖 체크)
            if x<0: w+=x; x=0
            if y<0: h+=y; y=0
            if x+w>bg_w: w=bg_w-x
            if y+h>bg_h: h=bg_h-y
            if w<=0 or h<=0: continue
            
            add_shadow(bg, x, y, w, h)
            
            fg_crop = fg_resized[:h, :w]
            bg_crop = bg[y:y+h, x:x+w]
            
            if fg_crop.shape[2] == 4:
                alpha = fg_crop[:, :, 3] / 255.0
                inv_alpha = 1.0 - alpha
                for c in range(3):
                    bg_crop[:, :, c] = (alpha * fg_crop[:, :, c] + inv_alpha * bg_crop[:, :, c])
                bg[y:y+h, x:x+w] = bg_crop
            else:
                bg[y:y+h, x:x+w] = fg_crop[:, :, :3]
                
            # 라벨 저장 (Normalized XYWH)
            real_cx = x + w/2
            real_cy = y + h/2
            labels.append(f"{class_map[c_name]} {real_cx/bg_w:.6f} {real_cy/bg_h:.6f} {w/bg_w:.6f} {h/bg_h:.6f}")

        # 파일 저장
        file_id = f"v4_{i:06d}"
        cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{file_id}.jpg", bg)
        with open(f"{OUTPUT_DIR}/labels/{split}/{file_id}.txt", "w") as f:
            f.write("\n".join(labels))

    print("\n" + "="*50)
    print(f"🎉 데이터 생성 완료! 총 {TOTAL_IMAGES}장")
    print(f"📂 저장 경로: {OUTPUT_DIR}")
    print(f"📝 data.yaml 생성 완료")
    print("="*50)

if __name__ == "__main__":
    generate_dataset_final()