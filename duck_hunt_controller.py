import cv2
import pyautogui
import webbrowser
import time
import numpy as np
import sys
from utils import HandGestureDetector

# Windows-specific for always-on-top window
try:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    HAS_WIN32 = True
except:
    HAS_WIN32 = False

def set_window_topmost(window_name):
    """Set OpenCV window to always stay on top (Windows only)."""
    if not HAS_WIN32:
        return False
    try:
        # Find the window handle
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            return True
    except:
        pass
    return False

def load_calibration():
    try:
        with open("calibration.txt", "r") as f:
            lines = f.readlines()
            extended_dist = float(lines[0].strip())
            folded_dist = float(lines[1].strip())
            return extended_dist, folded_dist
    except:
        return None, None

def ease_position(current, target, factor=0.25):
    """Exponential easing for smooth cursor movement."""
    diff = target - current
    if abs(diff) > 150:
        return target
    elif abs(diff) > 80:
        return current + diff * 0.6
    else:
        return current + diff * factor

def run_game_controller():
    extended_dist, folded_dist = load_calibration()
    
    if not extended_dist or not folded_dist:
        print("❌ Run calibration first: python calibration.py")
        return
    
    try:
        detector = HandGestureDetector(model_complexity=0)
        detector.set_trigger_distances(extended_dist, folded_dist)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Camera settings for low latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        screen_width, screen_height = pyautogui.size()
        
        print("Opening Duck Hunt...")
        webbrowser.open("https://duckhuntjs.com/")
        time.sleep(3)
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        print("\n" + "=" * 80)
        print("🔫 DUCK HUNT CONTROLLER ACTIVE 🔫")
        print("=" * 80)
        print("🖐️  Palm facing camera")
        print("👆 Index finger = Cursor")
        print("👍 Thumb EXTENDED = Ready")
        print("👊 Thumb TO PALM = Fire!")
        print("⌨️  'Q' quit, 'D' toggle display, 'R' reset, 'T' toggle topmost")
        print("=" * 80 + "\n")
        
        # Window setup
        window_name = "Duck Hunt Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 320, 240)
        
        # Move window to top-right corner
        cv2.moveWindow(window_name, screen_width - 340, 20)
        
        # Make window always on top
        time.sleep(0.5)  # Let window initialize
        topmost_enabled = set_window_topmost(window_name)
        if topmost_enabled:
            print("✓ Window set to always-on-top")
        else:
            print("⚠️  Could not set always-on-top (try running as admin)")
        
        # Screen position tracking
        last_screen_x = screen_width // 2
        last_screen_y = screen_height // 2
        
        show_display = True
        shot_count = 0
        frames_without_hand = 0
        was_firing = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            results = detector.detect_hands(frame)
            
            if results and results.multi_hand_landmarks:
                frames_without_hand = 0
                
                for hand_landmarks in results.multi_hand_landmarks:
                    barrel_pos = detector.get_barrel_position(hand_landmarks, w, h)
                    
                    # Map camera to screen
                    target_x = int(np.interp(barrel_pos[0], [0, w], [0, screen_width]))
                    target_y = int(np.interp(barrel_pos[1], [0, h], [0, screen_height]))
                    
                    # Apply easing
                    screen_x = int(ease_position(last_screen_x, target_x, 0.35))
                    screen_y = int(ease_position(last_screen_y, target_y, 0.35))
                    
                    pyautogui.moveTo(screen_x, screen_y, _pause=False)
                    last_screen_x, last_screen_y = screen_x, screen_y
                    
                    # Update trigger
                    detector.update_trigger_state(hand_landmarks)
                    is_firing = detector.is_trigger_pulled()
                    
                    if is_firing and not was_firing:
                        pyautogui.click(_pause=False)
                        shot_count += 1
                    
                    was_firing = is_firing
                    
                    if show_display:
                        current_dist = detector.get_current_distance(hand_landmarks)
                        raw_dist = detector.get_raw_distance(hand_landmarks)
                        
                        color = (0, 0, 255) if is_firing else (0, 255, 0)
                        
                        # Draw crosshair at barrel
                        cv2.circle(frame, barrel_pos, 12, color, 3)
                        cv2.line(frame, (barrel_pos[0]-15, barrel_pos[1]), 
                                (barrel_pos[0]+15, barrel_pos[1]), color, 2)
                        cv2.line(frame, (barrel_pos[0], barrel_pos[1]-15), 
                                (barrel_pos[0], barrel_pos[1]+15), color, 2)
                        
                        # Draw thumb position
                        thumb_tip = hand_landmarks.landmark[4]
                        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        thumb_color = (0, 0, 255) if is_firing else (255, 0, 255)
                        cv2.circle(frame, (tx, ty), 10, thumb_color, -1)
                        
                        # Status text
                        status = "FIRING!" if is_firing else "READY"
                        cv2.putText(frame, status, (10, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        
                        # Distance display
                        cv2.putText(frame, f"Dist: {current_dist:.3f} (raw: {raw_dist:.3f})", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Thresholds
                        cv2.putText(frame, 
                                   f"Fire<{detector.trigger_fire_dist:.3f} Release>{detector.trigger_release_dist:.3f}", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                        
                        # Shot counter
                        cv2.putText(frame, f"Shots: {shot_count}", (10, h-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                frames_without_hand += 1
                if frames_without_hand > 30:
                    detector.reset_filters()
                    frames_without_hand = 0
                
                if show_display:
                    cv2.putText(frame, "Show hand!", (10, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if show_display:
                # Resize for small overlay
                small_frame = cv2.resize(frame, (320, 240))
                cv2.imshow(window_name, small_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                show_display = not show_display
                if not show_display:
                    cv2.destroyAllWindows()
                else:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 320, 240)
                    cv2.moveWindow(window_name, screen_width - 340, 20)
                    time.sleep(0.3)
                    set_window_topmost(window_name)
            elif key == ord('r'):
                detector.reset_filters()
                print("  [Filters reset]")
            elif key == ord('t'):
                # Toggle topmost
                topmost_enabled = set_window_topmost(window_name)
                print(f"  [Topmost: {'enabled' if topmost_enabled else 'failed'}]")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n🎯 Total shots: {shot_count}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    run_game_controller()
