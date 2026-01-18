import cv2
import time
import numpy as np
import sys
from utils import HandGestureDetector

def run_calibration():
    print("=" * 80)
    print("🔫 DUCK HUNT - THUMB-TO-PALM CALIBRATION 🔫")
    print("=" * 80)
    print("\n💡 NEW GESTURE SYSTEM:\n")
    print("   🖐️  Palm facing camera")
    print("   👆 INDEX finger = Barrel (controls cursor)")
    print("   👍 Thumb EXTENDED outward = Ready (don't shoot)")
    print("   👊 Thumb TOUCHING palm = Fire!")
    print("\n   Distance shown: LARGE = ready, SMALL = fire\n")
    print("=" * 80)
    print("\nPress 'E' for thumb EXTENDED/ready (5 samples)")
    print("Press 'F' for thumb FOLDED/fire (5 samples)")
    print("Press 'Q' to quit\n")
    
    detector = HandGestureDetector()
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera!")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Camera opened")
        
        extended_samples = []
        folded_samples = []
        calibration_complete = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            results = detector.detect_hands(frame)
            if results:
                frame = detector.draw_landmarks(frame, results)
            
            # Header
            cv2.rectangle(frame, (0, 0), (w, 180), (40, 40, 40), -1)
            cv2.putText(frame, "THUMB-TO-PALM CALIBRATION", (w//2-240, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            if len(extended_samples) < 5:
                cv2.putText(frame, "1. Palm facing camera", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"2. Thumb EXTENDED out - Press 'E' ({len(extended_samples)}/5)", 
                           (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Should show LARGE distance (~0.25-0.35)", (10, 135), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 2)
            elif len(folded_samples) < 5:
                cv2.putText(frame, f"3. Thumb FOLDED to palm - Press 'F' ({len(folded_samples)}/5)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                cv2.putText(frame, "Should show SMALL distance (~0.08-0.15)", (10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 2)
            else:
                cv2.putText(frame, "CALIBRATION COMPLETE! Press 'Q' to start", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Current distance display
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    current_dist = detector.get_raw_distance(hand_landmarks)
                    
                    # Draw barrel position (index fingertip)
                    barrel_pos = detector.get_barrel_position(hand_landmarks, w, h)
                    cv2.circle(frame, barrel_pos, 12, (0, 255, 255), 2)
                    cv2.circle(frame, barrel_pos, 3, (0, 255, 255), -1)
                    
                    # Draw thumb tip
                    thumb_tip = hand_landmarks.landmark[4]
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    cv2.circle(frame, (tx, ty), 8, (255, 0, 255), -1)
                    cv2.putText(frame, "THUMB", (tx-25, ty-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    cv2.rectangle(frame, (0, h-90), (w, h), (40, 40, 40), -1)
                    
                    # Big distance display
                    dist_text = f"{current_dist:.3f}"
                    cv2.putText(frame, dist_text, (w//2-70, h-45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    cv2.putText(frame, "distance", (w//2-50, h-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    
                    if len(extended_samples) > 0:
                        ext_med = np.median(extended_samples)
                        cv2.putText(frame, f"EXT: {ext_med:.3f}", 
                                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if len(folded_samples) > 0:
                        fold_med = np.median(folded_samples)
                        cv2.putText(frame, f"FOLD: {fold_med:.3f}", 
                                   (180, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                cv2.putText(frame, "SHOW YOUR HAND!", (w//2-150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.imshow("Duck Hunt Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # EXTENDED position (thumb out)
            if key == ord('e') and results and results.multi_hand_landmarks and len(extended_samples) < 5:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Take 3 quick samples and use median
                    quick_samples = []
                    for _ in range(3):
                        quick_samples.append(detector.get_raw_distance(hand_landmarks))
                        time.sleep(0.05)
                    dist = np.median(quick_samples)
                    extended_samples.append(dist)
                    print(f"✓ EXTENDED sample {len(extended_samples)}/5: {dist:.3f} (should be large)")
                    time.sleep(0.2)
            
            # FOLDED position (thumb to palm)
            elif key == ord('f') and results and results.multi_hand_landmarks and len(extended_samples) == 5 and len(folded_samples) < 5:
                for hand_landmarks in results.multi_hand_landmarks:
                    quick_samples = []
                    for _ in range(3):
                        quick_samples.append(detector.get_raw_distance(hand_landmarks))
                        time.sleep(0.05)
                    dist = np.median(quick_samples)
                    folded_samples.append(dist)
                    print(f"✓ FOLDED sample {len(folded_samples)}/5: {dist:.3f} (should be small)")
                    
                    if len(folded_samples) == 5:
                        ext_median = np.median(extended_samples)
                        fold_median = np.median(folded_samples)
                        difference = abs(ext_median - fold_median)
                        
                        print(f"\n📊 Calibration Results:")
                        print(f"  Thumb EXTENDED: {ext_median:.3f}")
                        print(f"  Thumb FOLDED:   {fold_median:.3f}")
                        print(f"  Difference:     {difference:.3f}")
                        
                        if ext_median < fold_median:
                            print(f"\n⚠️  Swapping: extended should be > folded")
                            ext_median, fold_median = fold_median, ext_median
                        
                        if difference < 0.05:
                            print(f"\n❌ ERROR: Difference too small! ({difference:.3f} < 0.05)")
                            print("   Make thumb movements more extreme!")
                            response = input("   Restart? (y/n): ")
                            if response.lower() == 'y':
                                extended_samples.clear()
                                folded_samples.clear()
                                continue
                            else:
                                cap.release()
                                cv2.destroyAllWindows()
                                return False
                        
                        print(f"✓ Perfect! Difference: {difference:.3f}")
                        
                        with open("calibration.txt", "w") as f:
                            f.write(f"{ext_median}\n{fold_median}")
                        
                        print(f"\n🎮 Starting game in 3 seconds...\n")
                        calibration_complete = True
                        time.sleep(3)
                        break
            
            elif key == ord('q') or key == 27:
                if calibration_complete:
                    break
                else:
                    print("Calibration incomplete.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
        
        cap.release()
        cv2.destroyAllWindows()
        return calibration_complete
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return False

if __name__ == "__main__":
    try:
        success = run_calibration()
        if success:
            import os
            os.system("python duck_hunt_controller.py")
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
        sys.exit(0)
