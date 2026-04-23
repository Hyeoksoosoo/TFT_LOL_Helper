from ultralytics import YOLO
import torch
import os
import sys

# ==========================================
# ⚙️ [설정] RTX 4070 Ti + 3440x1440 최적화
# ==========================================

# ★ [중요] 방금 생성한 v4 (최종) 데이터셋 경로로 변경했습니다.
DATA_YAML_PATH = r"C:\TFT_helper\yolo_dataset_v4\data.yaml"

# 2. 프로젝트 저장 설정
PROJECT_NAME = "TFT_Project"
# 구분하기 쉽게 이름에 'final_v4'를 붙였습니다.
RUN_NAME = "set04_final_v4_data" 

def train_start():
    # ==========================================
    # 🔍 1. 하드웨어 가속 확인
    # ==========================================
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🔥 GPU 가동: {gpu_name}")
        print("   -> RTX 4070 Ti의 강력한 성능을 사용합니다.")
        device = 0
    else:
        print("\n⚠️ 경고: GPU를 찾을 수 없습니다! CPU로 실행하면 매우 느립니다.")
        device = 'cpu'

    # ==========================================
    # 🧠 2. 모델 로드
    # ==========================================
    print("⬇️ YOLO11 Large 모델 로딩 중...")
    try:
        model = YOLO('yolo11l.pt') 
    except Exception:
        print("⚠️ yolo11l.pt 로드 실패, yolov8l.pt로 대체합니다.")
        model = YOLO('yolov8l.pt')

    # ==========================================
    # 🚀 3. 학습 시작 (울트라와이드 + Augmentation)
    # ==========================================
    print("\n🚀 학습을 시작합니다! (모니터 해상도 3440 x 1440 대응)")
    print("   - 데이터셋: v4 (개별 튜닝 + 여백 제거 적용됨)")
    print("   - 해상도(imgsz): 1280 (선명도 확보)")
    print("   - 직사각형(rect): True (울트라와이드 비율)")
    print("   - 데이터 증강(Aug): On (색감변화, 모자이크, 회전 등)")
    
    model.train(
        data=DATA_YAML_PATH,
        
        # [기본 학습 설정]
        epochs=100,      
        patience=20,     # 성능 향상 없으면 20번 기다리다 종료
        
        # [메모리 설정]
        # RTX 4070 Ti (12GB) + Large 모델 + 1280 해상도는 메모리를 많이 먹습니다.
        # 만약 "CUDA out of memory" 에러가 뜨면 4로 줄이세요.
        batch=8,         
        
        imgsz=1280,      # 3440 해상도 대응을 위한 고해상도
        rect=True,       # 울트라와이드 비율 최적화
        
        # ==========================================
        # 🎨 [핵심] 실전 적응력 강화 (Augmentation)
        # ==========================================
        # 1. 색감/조명 변화 대응 (맵 스킨마다 다른 조명 극복)
        hsv_h=0.015,     # 색조
        hsv_s=0.5,       # 채도 (흑백~과포화)
        hsv_v=0.4,       # 명도 (그림자/밝은 빛)
        
        # 2. 위치/모양 변화 대응
        degrees=5.0,     # ±5도 회전
        translate=0.1,   # 위치 이동
        scale=0.2,       # 크기 변화
        shear=0.0,       
        flipud=0.0,      
        fliplr=0.5,      # 좌우 반전
        
        # 3. 모자이크 학습 (작은 객체 인식률 떡상 비기)
        # 사진 4장을 하나로 합쳐서 학습 -> 배경 의존도 낮춤
        mosaic=1.0,      
        
        # 기타 설정
        device=device,   
        workers=4,       
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,   
        plots=True,      
        amp=True         
    )

    print("\n" + "="*50)
    print(f"🎉 학습이 완료되었습니다!")
    print(f"💾 최종 모델 파일: runs/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print("   -> 이 파일을 사용하여 실전 인식을 수행하면 됩니다.")
    print("="*50)

if __name__ == '__main__':
    train_start()