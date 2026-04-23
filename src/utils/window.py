import win32gui
import win32con

def find_game_window(window_name="League of Legends (TM) Client"):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        # 가끔 이름이 다를 수 있어 일반적인 이름도 시도
        hwnd = win32gui.FindWindow(None, "League of Legends")
        
    if hwnd == 0:
        print("❌ 게임 창을 찾을 수 없습니다.")
        return None
        
    rect = win32gui.GetWindowRect(hwnd)
    x, y, w, h = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
    
    # 윈도우 테두리 등 오차 보정 (필요시 수정)
    return {'left': x, 'top': y, 'width': w, 'height': h}