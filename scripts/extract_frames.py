import cv2
import numpy as np
import os
import time
import mss
import sys

# =========================================================
# ⚙️ [설정] 저장 경로 및 옵션
# =========================================================
# 1. 이미지가 저장될 폴더 (새로 지정함)
OUTPUT_DIR = r"C:\TFT_helper\datasets\capture_raw"

# 2. 캡처 간격 (초 단위)
# 2초마다 한 장씩 찍습니다. (게임 한 판 30분 기준 약 900장 확보 가능)
INTERVAL_SEC = 2 

def capture_screen_loop():
    # 저장 폴더 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 [알림] 새 저장 폴더를 생성했습니다: {OUTPUT_DIR}")
    
    # 화면 캡처 도구 초기화
    sct = mss.mss()
    # 윈도우 데스크탑 메인 모니터 (3440 x 1440)
    monitor = sct.monitors[1] 

    print("=" * 50)
    print(f"📸 실시간 화면 캡처를 시작합니다.")
    print(f"⏱️  캡처 간격: {INTERVAL_SEC}초")
    print(f"📂 저장 경로: {OUTPUT_DIR}")
    print("🛑 종료하려면 콘솔에서 [Ctrl + C]를 누르세요.")
    print("=" * 50)

    count = 0

    try:
            start_time = time.time()

            # 1. 화면 캡처
            img_np = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR) # OpenCV 색상으로 변환

            # 2. 이미지 저장
            # 파일명 중복 방지를 위해 현재 시간(타임스탬프) 사용
            timestamp = int(time.time())
            filename = f"capture_{timestamp}_{count:04d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            cv2.imwrite(save_path, frame)
            count += 1

            # 3. 로그 출력
            sys.stdout.write(f"\r🚀 [{count}장] 저장됨: {filename}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print(f"🎉 캡처 종료!")
        print(f"✅ 총 {count}장의 이미지가 저장되었습니다.")
        print(f"📂 폴더 위치: {OUTPUT_DIR}")
        print("👉 이제 Roboflow에 업로드하여 라벨링을 시작하세요.")

if __name__ == "__main__":
    capture_screen_loop()