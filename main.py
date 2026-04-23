import time
import mss
import os
import sys
import cv2
import numpy as np

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from utils.window import find_game_window
from modules.status import StatusManager
from modules.items import ItemManager
from modules.shop import ShopRecognizer  # [NEW] 상점 모듈 추가

# =========================================================
# ⚙️ 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

IMG_PATH = os.path.join(DATA_DIR, "shop_raw")
CHAMP_JSON = os.path.join(DATA_DIR, "champions.json")
TRAIT_JSON = os.path.join(DATA_DIR, "traits.json")

def main():
    # 콘솔 청소 (Windows/Mac 호환)
    clear_cmd = 'cls' if os.name == 'nt' else 'clear'
    os.system(clear_cmd)

    print("🤖 [TFT AI] 통합 시스템 가동 중... (Ctrl+C로 종료)")
    print("🔄 모듈 로딩 중...")

    # 1. 매니저 초기화
    sct = mss.mss()
    status_tracker = StatusManager(sct)
    item_manager = ItemManager(sct)
    shop_tracker = ShopRecognizer(IMG_PATH, CHAMP_JSON, TRAIT_JSON) # [NEW]

    print("✅ 모든 시스템 준비 완료! 게임을 감시합니다.")
    time.sleep(1)

    try:
        while True:
            start_time = time.time()
            window_rect = find_game_window()
            
            if window_rect:
                # ---------------------------------------------------------
                # 0. 화면 캡처 (상점 인식을 위해 전체 화면 필요)
                # ---------------------------------------------------------
                # mss로 모니터 전체 캡처 (상점 좌표가 전체 화면 기준이므로)
                monitor = sct.monitors[1] 
                full_img_np = np.array(sct.grab(monitor))
                full_img = cv2.cvtColor(full_img_np, cv2.COLOR_BGRA2BGR)

                # ---------------------------------------------------------
                # 1. 상태 정보 수집 (API + OCR)
                # ---------------------------------------------------------
                player_info = status_tracker.get_player_info()
                gold = status_tracker.get_gold(window_rect)
                curr_xp, max_xp, raw_xp_text = status_tracker.get_xp(window_rect, player_info['level'])
                
                # ---------------------------------------------------------
                # 2. 아이템 수집
                # ---------------------------------------------------------
                captured_slots = item_manager.capture_slots(window_rect)
                my_items = []
                for img in captured_slots:
                    name = item_manager.identify_item(img)
                    if name:
                        my_items.append(name)
                
                # 아이템 없으면 빈 리스트 처리
                if not my_items:
                    my_items_str = "없음"
                else:
                    my_items_str = ", ".join(my_items)

                # ---------------------------------------------------------
                # 3. [NEW] 상점 인식
                # ---------------------------------------------------------
                shop_list = shop_tracker.recognize(full_img)

                # ---------------------------------------------------------
                # 4. 통합 대시보드 출력
                # ---------------------------------------------------------
                os.system(clear_cmd) # 화면을 지워서 깔끔하게 갱신
                
                # 헤더
                print(f"⏱️ Update: {time.strftime('%H:%M:%S')} (Lat: {time.time() - start_time:.2f}s)")
                print("="*60)
                
                # 플레이어 상태
                xp_str = f"{curr_xp}/{max_xp}" if curr_xp != -1 else f"ERR({raw_xp_text})"
                print(f"👤 플레이어 : {player_info['name']}")
                print(f"📊 레벨     : Lv.{player_info['level']} ({xp_str})")
                print(f"💰 골드     : {gold}G")
                print(f"🎒 아이템   : {my_items_str}")
                
                print("-" * 60)
                print("🛒 [현재 상점]")
                
                # 상점 목록 출력 (한국어 + 시너지)
                for item in shop_list:
                    print(shop_tracker.get_display_info(item))
                
                print("="*60)

            else:
                print(f"\r❌ 게임 창을 찾을 수 없습니다... (League of Legends)", end="")
                
            # 너무 빠른 반복 방지 (CPU 절약)
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()