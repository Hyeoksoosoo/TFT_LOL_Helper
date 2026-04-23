from ultralytics import YOLO
import os
import shutil
import random
import glob
import sys

# ==========================================
# ⚙️ [수정됨] 절대 경로 설정 (버그 방지)
# ==========================================
# 현재 파일(train_classifier.py)의 위치: C:/TFT_helper
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 루트: C:/TFT_helper/datasets
DATASET_ROOT = os.path.join(CURRENT_DIR, "datasets")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")

EPOCHS = 10
IMG_SIZE = 64

def check_and_split_dataset():
    """데이터셋이 진짜 있는지 확인하고, Train/Val로 나눕니다."""
    print(f"📂 데이터셋 경로 확인 중: {DATASET_ROOT}")
    
    # 1. 폴더 존재 여부 확인
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 오류: '{TRAIN_DIR}' 폴더가 없습니다.")
        print("👉 'generate_dataset.py'를 먼저 실행해서 데이터를 만들어주세요!")
        return False

    # 2. 내용물(챔피언 폴더) 확인
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if len(classes) == 0:
        print(f"❌ 오류: '{TRAIN_DIR}' 폴더가 비어있습니다! (총 0개 클래스)")
        print("👉 데이터 생성이 제대로 안 됐습니다. 'generate_dataset.py'를 다시 실행하세요.")
        return False

    print(f"✅ 학습 데이터 발견: 총 {len(classes)}개 챔피언 클래스")

    # 3. Val 폴더가 비어있으면 데이터 분할 수행
    if not os.path.exists(VAL_DIR) or len(os.listdir(VAL_DIR)) == 0:
        print("🔄 검증(Val) 데이터셋이 없어서 생성합니다...")
        os.makedirs(VAL_DIR, exist_ok=True)
        
        for cls in classes:
            src_path = os.path.join(TRAIN_DIR, cls)
            dst_path = os.path.join(VAL_DIR, cls)
            
            os.makedirs(dst_path, exist_ok=True)
            images = glob.glob(os.path.join(src_path, "*.jpg"))
            
            # 20% 이동
            num_val = int(len(images) * 0.2)
            val_images = random.sample(images, num_val)
            
            for img in val_images:
                shutil.move(img, os.path.join(dst_path, os.path.basename(img)))
        print("✅ 데이터 분할 완료!")
    else:
        print("✅ 검증(Val) 데이터셋이 이미 존재합니다. 스킵.")
        
    return True

def train_model():
    print(f"🚀 학습 시작! (Epochs: {EPOCHS})")
    
    # 모델 로드
    model = YOLO('yolov8n-cls.pt') 
    
    # 학습 시작 (절대 경로 사용)
    results = model.train(
        data=DATASET_ROOT, 
        epochs=EPOCHS, 
        imgsz=IMG_SIZE, 
        project=os.path.join(CURRENT_DIR, 'tft_model'), 
        name='champion_classifier',
        plots=True
    )
    
    print("\n" + "="*50)
    print("🎉 학습 완료!")
    print(f"💾 모델 저장 경로: {os.path.join(CURRENT_DIR, 'tft_model', 'champion_classifier', 'weights', 'best.pt')}")
    print("="*50)

if __name__ == "__main__":
    if check_and_split_dataset():
        train_model()
    else:
        print("\n🛑 학습을 중단합니다. 위 오류 메시지를 확인하세요.")