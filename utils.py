import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class OneEuroFilter:
    """1€ Filter for smooth, low-latency signal filtering."""
    def __init__(self, min_cutoff=1.0, beta=0.007):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            dt = 0.001
        
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(dt, 1.0)
        dx_smooth = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        alpha = self._smoothing_factor(dt, cutoff)
        x_smooth = alpha * x + (1 - alpha) * self.x_prev
        
        self.x_prev = x_smooth
        self.dx_prev = dx_smooth
        self.t_prev = t
        
        return x_smooth
    
    def _smoothing_factor(self, dt, cutoff):
        r = 2 * 3.14159 * cutoff * dt
        return r / (r + 1)
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

class HandGestureDetector:
    """
    Gesture detector using THUMB-TO-PALM DISTANCE for trigger detection.
    
    New gesture system:
    - Palm facing camera, index finger = barrel (cursor)
    - Thumb EXTENDED outward = READY (don't shoot) - LARGE distance
    - Thumb TOUCHING/CROSSING palm = FIRE - SMALL distance
    """
    
    def __init__(self, model_complexity=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Landmarks
        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.PINKY_MCP = 17
        
        # Cursor smoothing - tuned for responsive gaming
        self.filter_x = OneEuroFilter(min_cutoff=2.5, beta=0.015)
        self.filter_y = OneEuroFilter(min_cutoff=2.5, beta=0.015)
        
        # THUMB-TO-PALM DISTANCE TRIGGER
        self.trigger_state = False
        self.thumb_extended_dist = None   # Large distance (ready)
        self.thumb_folded_dist = None     # Small distance (fire)
        self.trigger_fire_dist = None     # Below this = fire
        self.trigger_release_dist = None  # Above this = release
        self.last_shot_time = 0
        self.shot_cooldown = 0.30  # Fast shooting
        
        # Distance smoothing and history
        self.distance_history = deque(maxlen=5)
        self.distance_filter = OneEuroFilter(min_cutoff=1.5, beta=0.2)
        self.last_valid_distance = 0.15
        
        # Confirmation counters
        self.fire_confirm_count = 0
        self.release_confirm_count = 0
        self.required_fire_confirms = 2
        self.required_release_confirms = 2
        
    def detect_hands(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            return results
        except:
            return None
    
    def draw_landmarks(self, frame, results):
        try:
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        except:
            pass
        return frame
    
    def get_barrel_position(self, hand_landmarks, frame_width, frame_height):
        """Get index fingertip position as barrel/cursor."""
        try:
            index_tip = hand_landmarks.landmark[self.INDEX_TIP]
            
            pixel_x = int(index_tip.x * frame_width)
            pixel_y = int(index_tip.y * frame_height)
            
            t = time.time()
            smoothed_x = self.filter_x(float(pixel_x), t)
            smoothed_y = self.filter_y(float(pixel_y), t)
            
            return (int(smoothed_x), int(smoothed_y))
        except:
            return (frame_width // 2, frame_height // 2)
    
    def calculate_thumb_palm_distance(self, hand_landmarks):
        """
        Calculate normalized distance from thumb tip to palm center.
        
        Palm center is approximated as center of INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP.
        Distance is normalized by hand size (wrist to middle fingertip).
        
        Returns:
        - LARGE value (~0.25-0.35): Thumb extended = READY
        - SMALL value (~0.05-0.15): Thumb touching palm = FIRE
        """
        try:
            thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
            
            # Calculate palm center (average of knuckle positions)
            index_mcp = hand_landmarks.landmark[self.INDEX_MCP]
            middle_mcp = hand_landmarks.landmark[self.MIDDLE_MCP]
            ring_mcp = hand_landmarks.landmark[self.RING_MCP]
            pinky_mcp = hand_landmarks.landmark[self.PINKY_MCP]
            
            palm_x = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
            palm_y = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4
            palm_z = (index_mcp.z + middle_mcp.z + ring_mcp.z + pinky_mcp.z) / 4
            
            # Calculate distance in 3D
            dx = thumb_tip.x - palm_x
            dy = thumb_tip.y - palm_y
            dz = thumb_tip.z - palm_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Normalize by hand size (wrist to middle fingertip)
            wrist = hand_landmarks.landmark[self.WRIST]
            middle_tip = hand_landmarks.landmark[self.MIDDLE_TIP]
            hand_size = np.sqrt(
                (middle_tip.x - wrist.x)**2 + 
                (middle_tip.y - wrist.y)**2 + 
                (middle_tip.z - wrist.z)**2
            )
            
            if hand_size < 0.01:
                return self.last_valid_distance
            
            normalized_dist = distance / hand_size
            self.last_valid_distance = normalized_dist
            
            return normalized_dist
        except:
            return self.last_valid_distance
    
    def get_smoothed_distance(self, raw_distance):
        """Apply temporal smoothing with outlier rejection."""
        self.distance_history.append(raw_distance)
        
        if len(self.distance_history) >= 3:
            median = np.median(list(self.distance_history))
            std = np.std(list(self.distance_history))
            
            # Reject outliers
            if abs(raw_distance - median) > 3 * max(std, 0.02):
                return median
        
        return self.distance_filter(raw_distance)
    
    def set_trigger_distances(self, extended_dist, folded_dist):
        """
        Set trigger thresholds based on calibration.
        extended_dist: distance when thumb extended (large value = ready)
        folded_dist: distance when thumb folded (small value = fire)
        """
        # Ensure extended > folded (extended is larger)
        if extended_dist < folded_dist:
            print(f"⚠️  Swapping distances: extended should be > folded")
            extended_dist, folded_dist = folded_dist, extended_dist
        
        self.thumb_extended_dist = extended_dist
        self.thumb_folded_dist = folded_dist
        
        dist_range = extended_dist - folded_dist
        
        # Fire when distance drops below this (thumb moves toward palm)
        self.trigger_fire_dist = folded_dist + dist_range * 0.40
        
        # Release when distance goes above this (thumb extends out)
        self.trigger_release_dist = folded_dist + dist_range * 0.60
        
        print(f"\n🎯 Thumb-to-Palm Distance Trigger:")
        print(f"  Thumb EXTENDED (ready): {extended_dist:.3f} (large)")
        print(f"  Thumb FOLDED (fire):    {folded_dist:.3f} (small)")
        print(f"  Distance range:         {dist_range:.3f}")
        print(f"  Fire when BELOW:        {self.trigger_fire_dist:.3f}")
        print(f"  Release when ABOVE:     {self.trigger_release_dist:.3f}")
        print(f"  Confirmation frames:    {self.required_fire_confirms}")
    
    def update_trigger_state(self, hand_landmarks):
        """
        Update trigger based on thumb-to-palm distance.
        Fire when distance DECREASES (thumb moves to palm).
        Release when distance INCREASES (thumb extends out).
        """
        if self.trigger_fire_dist is None:
            return False
        
        raw_distance = self.calculate_thumb_palm_distance(hand_landmarks)
        smoothed_dist = self.get_smoothed_distance(raw_distance)
        current_time = time.time()
        
        if not self.trigger_state:
            # Not firing - check if distance decreased (thumb folded toward palm)
            if smoothed_dist < self.trigger_fire_dist:
                self.fire_confirm_count += 1
                if self.fire_confirm_count >= self.required_fire_confirms:
                    if (current_time - self.last_shot_time) > self.shot_cooldown:
                        self.trigger_state = True
                        self.last_shot_time = current_time
                        self.fire_confirm_count = 0
                        print(f"  [FIRE! dist={smoothed_dist:.3f} (raw={raw_distance:.3f})]")
            else:
                self.fire_confirm_count = max(0, self.fire_confirm_count - 1)
        else:
            # Firing - check if distance increased (thumb extended)
            if smoothed_dist > self.trigger_release_dist:
                self.release_confirm_count += 1
                if self.release_confirm_count >= self.required_release_confirms:
                    self.trigger_state = False
                    self.release_confirm_count = 0
                    print(f"  [READY dist={smoothed_dist:.3f} (raw={raw_distance:.3f})]")
            else:
                self.release_confirm_count = max(0, self.release_confirm_count - 1)
        
        return self.trigger_state
    
    def is_trigger_pulled(self):
        return self.trigger_state
    
    def get_current_distance(self, hand_landmarks):
        """Get current thumb-palm distance (smoothed) for display."""
        raw = self.calculate_thumb_palm_distance(hand_landmarks)
        return self.get_smoothed_distance(raw)
    
    def get_raw_distance(self, hand_landmarks):
        """Get raw distance without smoothing (for calibration)."""
        return self.calculate_thumb_palm_distance(hand_landmarks)
    
    def reset_filters(self):
        """Reset all filters and state."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.distance_filter.reset()
        self.distance_history.clear()
        self.fire_confirm_count = 0
        self.release_confirm_count = 0
        self.trigger_state = False
