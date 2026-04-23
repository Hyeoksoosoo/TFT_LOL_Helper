import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time

# ==========================================
# ⚙️ 설정
# ==========================================
# ★ [수정] 방금 학습시킨 '최종 v4' 모델 경로로 변경
# (학습이 아직 안 끝났다면 파일이 없으니, 학습 완료 후 실행하세요!)
MODEL_PATH = r"C:\TFT_helper\TFT_Project\set04_final_v4_data\weights\best.pt"

# 듀얼 모니터 설정 (1: 주모니터, 2: 보조모니터)
MONITOR_IDX = 1 

def run_inference():
    # 1. 모델 로딩
    print(f"⬇️ 모델 로딩 중... ({MODEL_PATH})")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 파일이 아직 없습니다! 학습이 끝나면 실행해주세요.\n{e}")
        return

    # 2. 화면 캡처 준비
    sct = mss.mss()
    if len(sct.monitors) <= MONITOR_IDX:
        print(f"❌ 모니터 인덱스 {MONITOR_IDX}번 오류 (현재 모니터 {len(sct.monitors)-1}개)")
        return
        
    monitor = sct.monitors[MONITOR_IDX]
    print(f"🚀 AI 가동 시작! (해상도: {monitor['width']}x{monitor['height']})")
    print("   -> 종료하려면 화면 클릭 후 'q'를 누르세요.")

    prev_time = 0
    
    while True:
        # 스크린샷 찍기
        img_np = np.array(sct.grab(monitor))
        
        # 색상 변환 (BGRA -> BGR)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        # 3. AI 예측
        # imgsz=1280 : 학습 환경과 동일하게 설정 (중요)
        # conf=0.5   : 50% 이상 확실할 때만 표시
        # (만약 인식이 잘 안 되면 0.4로 낮추고, 엉뚱한 게 잡히면 0.6으로 올리세요)
        results = model.predict(frame, imgsz=1280, conf=0.5, verbose=False)

        # 4. 결과 그리기
        # line_width=2: 박스 두께
        # font_size=1.5: 3440 해상도라 글씨를 좀 키웠습니다
        annotated_frame = results[0].plot(line_width=2, font_size=1.5)

        # 5. FPS 계산 및 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # FPS 텍스트 (좌측 상단)
        cv2.putText(annotated_frame, f"FPS: {int(fps)} (RTX 4070 Ti)", (40, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        # 6. 미리보기 창 크기 조절
        # 3440 원본은 너무 크니까 보기 편하게 절반 정도로 줄임
        view_width = 1720 
        scale = view_width / annotated_frame.shape[1]
        dim = (view_width, int(annotated_frame.shape[0] * scale))
        resized_frame = cv2.resize(annotated_frame, dim, interpolation=cv2.INTER_AREA)

        # 7. 출력
        cv2.imshow("TFT AI Monitor V4", resized_frame)

        # 'q' 키 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()