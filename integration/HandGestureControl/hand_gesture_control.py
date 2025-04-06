import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
import math
import pyttsx3
import os
import logging
from collections import deque
import threading
import warnings

class HandGestureControl:
    def __init__(self, show_window=True):
        # First initialize the absolute minimum required attributes
        self._initialized = False
        self.running = False
        self.show_window = show_window
        
        # Initialize logging before anything else
        self._initialize_logging()
        self.logger.info("Starting HandGestureControl initialization")
        
        try:
            # Suppress MediaPipe warnings
            self._suppress_warnings()
            
            # Initialize all components
            self._initialize_components()
            
            self._initialized = True
            self.logger.info("HandGestureControl initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self._safe_cleanup()
            raise

    def _initialize_logging(self):
        """Initialize logging infrastructure that will be available immediately"""
        self.logger = logging.getLogger('HandGestureControl')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _suppress_warnings(self):
        """Suppress MediaPipe and other warnings"""
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger('mediapipe').setLevel(logging.WARNING)

    def _initialize_components(self):
        """Initialize all the components of the hand gesture control"""
        # Constants and Settings
        self.wCam, self.hCam = 1280, 720
        self.frameR = 100  # Frame Reduction for mouse
        self.smoothening = 5
        self.min_click_distance = 40
        self.scroll_threshold = 25
        self.scroll_sensitivity = 1.5
        self.min_scroll_distance = 15
        self.click_hold_time = 0.3
        self.action_cooldown = 1.0
        self.SCREENSHOT_FOLDER = "screenshots"
        os.makedirs(self.SCREENSHOT_FOLDER, exist_ok=True)

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # Initialize camera
        self._initialize_camera()

        # Hand detector setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.wScr, self.hScr = pyautogui.size()

        # State variables
        self.current_mode = "mouse"  # mouse, udrl, ccpss, special
        self.last_mode_change = 0
        self.mode_cooldown = 0.5
        pyautogui.FAILSAFE = False

        # Mouse control variables
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.scroll_active = False
        self.cursor_locked = False
        self.left_click_active = False
        self.right_click_active = False
        self.last_right_click_time = 0
        self.right_click_cooldown = 0.3
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.01
        self.last_thumb_y = 0
        self.cursor_lock_pos = (0, 0)
        self.scroll_reference_y = 0
        self.click_start_time = 0
        self.is_click_hold = False

        # UDRL variables
        self.last_press_time = 0
        self.udrl_cooldown = 0.4
        self.last_fist_time = 0
        self.fist_cooldown = 0.8
        self.fist_frames = 0
        self.fist_confidence = 2
        self.prev_x, self.prev_y = None, None

        # CCPSS variables
        self.last_action_time = 0

        # Special mode variables
        self.tts_active = False
        self.last_gesture_time = 0
        self.gesture_buffer = deque(maxlen=5)

        # Finger tip IDs
        self.tipIds = [4, 8, 12, 16, 20]

    def _initialize_camera(self):
        """Initialize camera with proper error handling"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(3, self.wCam)
            self.cap.set(4, self.hCam)
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            raise

    def _safe_cleanup(self):
        """Safe cleanup that can be called during failed initialization"""
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands'):
                self.hands.close()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during safe cleanup: {str(e)}")
    def fingers_up(self, hand_landmarks):
        fingers = []
        
        # Thumb (compare x-coordinate of tip to base)
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (compare y-coordinate of tip to base)
        for id in range(1, 5):
            if hand_landmarks.landmark[self.tipIds[id]].y < hand_landmarks.landmark[self.tipIds[id]-2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def find_distance(self, p1, p2, landmarks, img=None, draw=True):
        x1, y1 = int(landmarks.landmark[p1].x * self.wCam), int(landmarks.landmark[p1].y * self.hCam)
        x2, y2 = int(landmarks.landmark[p2].x * self.wCam), int(landmarks.landmark[p2].y * self.hCam)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, (x1, y1, x2, y2, cx, cy)

    def is_fist(self, landmarks):
        # Check if fingertips are close to palm center
        palm_center_x = (landmarks.landmark[0].x + landmarks.landmark[9].x) / 2
        palm_center_y = (landmarks.landmark[0].y + landmarks.landmark[9].y) / 2
        
        distances = []
        for tip in [4, 8, 12, 16, 20]:  # All fingertips
            dx = landmarks.landmark[tip].x - palm_center_x
            dy = landmarks.landmark[tip].y - palm_center_y
            distances.append(math.hypot(dx, dy) * 1000)  # Scale up
        
        return all(d < 80 for d in distances)

    def handle_mouse_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Optimize by caching frequently used landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        # 1. Cursor Lock and Scroll Mode (thumb down)
        if fingers[0] == 1:  # Thumb down
            thumb_index_dist, _ = self.find_distance(4, 8, landmarks, img)
            
            if thumb_index_dist < self.scroll_threshold:  # SCROLL MODE
                if not self.scroll_active:
                    self.scroll_active = True
                    self.cursor_locked = True
                    self.cursor_lock_pos = (self.clocX, self.clocY)
                    self.scroll_reference_y = int(thumb_tip.y * self.hCam)
                    self.last_thumb_y = self.scroll_reference_y
                
                # Lock cursor position
                if self.cursor_lock_pos:
                    pyautogui.moveTo(*self.cursor_lock_pos)
                
                # Calculate scroll amount with deadzone and acceleration
                current_thumb_y = int(thumb_tip.y * self.hCam)
                vertical_movement = self.scroll_reference_y - current_thumb_y
                
                if abs(vertical_movement) > self.min_scroll_distance and (current_time - self.last_scroll_time) > self.scroll_cooldown:
                    # Add acceleration based on movement speed
                    scroll_factor = min(1.0, abs(vertical_movement) / (self.hCam * 0.1))  # Normalize by screen height
                    scroll_amount = int(scroll_factor * vertical_movement * self.scroll_sensitivity)
                    
                    pyautogui.scroll(scroll_amount)
                    self.last_scroll_time = current_time
                    self.scroll_reference_y = current_thumb_y
                    
                    # Only update UI if needed
                    if img is not None:
                        direction = "Scroll Up" if scroll_amount > 0 else "Scroll Down"
                        cv2.putText(img, direction, (40, 150), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            else:  # Regular cursor lock
                if not self.cursor_locked:
                    self.cursor_locked = True
                    self.cursor_lock_pos = (self.clocX, self.clocY)
                if self.cursor_lock_pos:
                    pyautogui.moveTo(*self.cursor_lock_pos)
                if img is not None:
                    cv2.putText(img, "Cursor Locked", (40, 150), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 0), 2)
                self.scroll_active = False
        else:  # Thumb up
            self.scroll_active = False
            self.cursor_locked = False
        
        # 2. Mouse Movement (index finger up, thumb up) - Optimized with prediction
        if fingers[1] == 1 and fingers[0] == 0:
            # Convert coordinates with boundary checks
            x1 = max(self.frameR, min(self.wCam - self.frameR, int(index_tip.x * self.wCam)))
            y1 = max(self.frameR, min(self.hCam - self.frameR, int(index_tip.y * self.hCam)))
            
            # Smoother interpolation with prediction
            target_x = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
            target_y = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
            
            # Predictive smoothing with velocity factor
            vel_x = (target_x - self.plocX) * 0.5  # Velocity component (adjust factor as needed)
            vel_y = (target_y - self.plocY) * 0.5
            
            self.clocX = self.plocX + (target_x - self.plocX + vel_x) / self.smoothening
            self.clocY = self.plocY + (target_y - self.plocY + vel_y) / self.smoothening
            
            # Move mouse (only if significant movement)
            if abs(self.clocX - self.plocX) > 1 or abs(self.clocY - self.plocY) > 1:
                pyautogui.moveTo(self.clocX, self.clocY)
                if img is not None:
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            self.plocX, self.plocY = self.clocX, self.clocY
        
        # 3. Left Click - Optimized with debouncing
        if fingers[0] == 0:  # Thumb up
            if fingers[1] == 1 and fingers[2] == 1:  # Index and middle up
                length, _ = self.find_distance(8, 12, landmarks, img)
                
                if length < self.min_click_distance:
                    if not self.left_click_active:
                        self.left_click_active = True
                        self.click_start_time = current_time
                        self.is_click_hold = False
                    
                    # Hold detection with visual feedback
                    if current_time - self.click_start_time > self.click_hold_time and not self.is_click_hold:
                        pyautogui.mouseDown(button='left')
                        self.is_click_hold = True
                        if img is not None:
                            cv2.putText(img, "Press-to-Select", (40, 200), 
                                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                elif self.left_click_active:
                    # Debounce release
                    if current_time - self.click_start_time > 0.05:  # 50ms debounce
                        if not self.is_click_hold:  # Single click
                            pyautogui.click(button='left')
                        else:  # Release press-to-select
                            pyautogui.mouseUp(button='left')
                        self.left_click_active = False
            elif self.left_click_active and self.is_click_hold:
                pyautogui.mouseUp(button='left')
                self.left_click_active = False
        
        # 4. Right Click - Optimized with cooldown
        if fingers[0] == 0 and not self.left_click_active:  # Thumb up and no left click active
            if fingers[4] == 1 and fingers[2] == 0:  # Pinky up, middle down
                if not self.right_click_active and (current_time - self.last_right_click_time) > self.right_click_cooldown:
                    pyautogui.click(button='right')
                    self.right_click_active = True
                    self.last_right_click_time = current_time
                    if img is not None:
                        cv2.putText(img, "Right Click!", (40, 200), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                self.right_click_active = False

    def handle_udrl_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Fist detection for ENTER
        if self.is_fist(landmarks):
            self.fist_frames += 1
            cv2.putText(img, "FIST DETECTED", (200, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            if self.fist_frames >= self.fist_confidence and current_time - self.last_fist_time > self.fist_cooldown:
                pyautogui.press("enter")
                self.last_fist_time = current_time
                self.fist_frames = 0
            return
        
        self.fist_frames = 0
        
        # Get index finger tip position
        index_tip = landmarks.landmark[8]
        current_x, current_y = int(index_tip.x * self.wCam), int(index_tip.y * self.hCam)
        
        cv2.circle(img, (current_x, current_y), 12, (0, 255, 0), -1)
        
        if self.prev_x is None:
            self.prev_x, self.prev_y = current_x, current_y
        
        delta_x = current_x - self.prev_x
        delta_y = current_y - self.prev_y
        distance = math.hypot(delta_x, delta_y)
        
        # Get direction
        direction = None
        color = (255, 255, 255)
        
        delta_y = -delta_y  # Invert y-axis
        angle = math.degrees(math.atan2(delta_y, delta_x))
        
        if -45 <= angle < 45 and distance > 60:  # Right
            direction = "right"
            color = (255, 0, 0)
        elif 45 <= angle < 135 and distance > 50:  # Up
            direction = "up"
            color = (0, 255, 0)
        elif (angle >= 135 or angle < -135) and distance > 80:  # Left
            direction = "left"
            color = (0, 0, 255)
        elif -135 <= angle < -45 and distance > 100:  # Down
            direction = "down"
            color = (0, 255, 255)
        
        if direction and distance > 30:
            cv2.line(img, (self.prev_x, self.prev_y), (current_x, current_y), color, 4)
            
            if current_time - self.last_press_time > self.udrl_cooldown:
                pyautogui.press(direction)
                self.last_press_time = current_time
        
        self.prev_x, self.prev_y = current_x, current_y

    def handle_ccpss_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Copy (thumb up only)
        if fingers == [1, 0, 0, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'c')
            self.last_action_time = current_time
            cv2.putText(img, "COPY", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        # Cut (index + middle up)
        elif fingers == [0, 1, 1, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'x')
            self.last_action_time = current_time
            cv2.putText(img, "CUT", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        # Paste (thumb + index up)
        elif fingers == [1, 1, 0, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'v')
            self.last_action_time = current_time
            cv2.putText(img, "PASTE", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Screenshot (all fingers up)
        elif fingers == [1, 1, 1, 1, 1] and current_time - self.last_action_time > self.action_cooldown:
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.SCREENSHOT_FOLDER, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, screenshot)
            self.last_action_time = current_time
            cv2.putText(img, "SCREENSHOT", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    def handle_special_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Check for activation (index + middle up)
        if fingers[1] and fingers[2] and not any(fingers[3:]):
            self.gesture_buffer.append("activate")
        
        # Check for deactivation (no fingers up)
        elif not any(fingers[1:]):
            self.gesture_buffer.append("deactivate")
        else:
            self.gesture_buffer.append("none")
        
        # Check buffer for consistent gesture
        if current_time - self.last_gesture_time > self.action_cooldown:
            if self.gesture_buffer.count("activate") >= 3 and not self.tts_active:
                self.engine.say("Speech to text activated")
                self.engine.runAndWait()
                pyautogui.hotkey('win', 'h')
                self.tts_active = True
                self.last_gesture_time = current_time
            
            elif self.gesture_buffer.count("deactivate") >= 3 and self.tts_active:
                self.engine.say("Speech to text deactivated")
                self.engine.runAndWait()
                pyautogui.hotkey('win', 'h')
                self.tts_active = False
                self.last_gesture_time = current_time
        
        # Display status
        status_color = (0, 255, 0) if self.tts_active else (0, 0, 255)
        cv2.putText(img, f"TTS: {'ON' if self.tts_active else 'OFF'}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    def detect_mode(self, left_hand_landmarks):
        if not left_hand_landmarks:
            return "mouse"
        
        fingers = self.fingers_up(left_hand_landmarks)
        count = sum(fingers[1:])  # Exclude thumb
        
        if count == 1 and fingers[1] == 1:  # Index finger up
            return "udrl"
        elif count == 2 and fingers[1] == 1 and fingers[2] == 1:  # Index + middle
            return "ccpss"
        elif count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:  # Index + middle + ring
            return "special"
        return "mouse"
    def run(self):
        self.running = True
        pTime = 0
        
        try:
            while self.running:
                success, img = self.cap.read()
                if not success:
                    self.logger.warning("Failed to capture frame")
                    continue

                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                left_hand = None
                right_hand = None
                
                # Identify hands
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_type = results.multi_handedness[hand_idx].classification[0].label
                        if hand_type == "Left":
                            left_hand = hand_landmarks
                        else:
                            right_hand = hand_landmarks
                        
                        # Draw hand landmarks
                        if self.show_window:
                            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect mode from left hand
                new_mode = self.detect_mode(left_hand)
                if new_mode != self.current_mode and (time.time() - self.last_mode_change) > self.mode_cooldown:
                    self.current_mode = new_mode
                    self.last_mode_change = time.time()
                    self.logger.info(f"Mode changed to: {self.current_mode}")
                
                # Process right hand based on current mode
                if right_hand:
                    if self.current_mode == "mouse":
                        self.handle_mouse_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "udrl":
                        self.handle_udrl_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "ccpss":
                        self.handle_ccpss_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "special":
                        self.handle_special_mode(right_hand, img if self.show_window else None)
                
                # Only show UI if window is enabled
                if self.show_window:
                    # Display mode information
                    mode_colors = {
                        "mouse": (0, 255, 255),
                        "udrl": (255, 0, 0),
                        "ccpss": (0, 255, 0),
                        "special": (255, 0, 255)
                    }
                    cv2.putText(img, f"MODE: {self.current_mode.upper()}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, mode_colors[self.current_mode], 2)
                    
                    # Display instructions
                    cv2.putText(img, "Left Hand Modes:", (10, self.hCam-120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    cv2.putText(img, "No fingers: Mouse | 1 finger: UDRL", (10, self.hCam-90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    cv2.putText(img, "2 fingers: CCPSS | 3 fingers: Special", (10, self.hCam-60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    
                    # Display FPS
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(img, f"FPS: {int(fps)}", (self.wCam-150, 50), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                    cv2.imshow("Integrated Hand Control", img)
                
                # Check for quit command (even if window is hidden)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

        except Exception as e:
            self.logger.error(f"Error in hand gesture control: {str(e)}")
        finally:
            self.stop()

    def start(self):
        """Start the hand gesture control in a separate thread"""
        if not self.running:
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Hand gesture control started")

    def stop(self):
        """Stop the hand gesture control and clean up resources"""
        if not hasattr(self, '_initialized') or not self._initialized:
            return

        self.running = False
        
        try:
            if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'show_window') and self.show_window:
                cv2.destroyAllWindows()
            self.logger.info("Hand gesture control stopped")
        except Exception as e:
            self.logger.error(f"Error during stop: {str(e)}")

    def __del__(self):
        """Destructor to ensure resources are released"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info("Cleaning up HandGestureControl resources")
            self.stop()
        except Exception:
            pass
class HandGestureControl:
    def __init__(self, show_window=True):
        # First initialize the absolute minimum required attributes
        self._initialized = False
        self.running = False
        self.show_window = show_window
        
        # Initialize logging before anything else
        self._initialize_logging()
        self.logger.info("Starting HandGestureControl initialization")
        
        try:
            # Suppress MediaPipe warnings
            self._suppress_warnings()
            
            # Initialize all components
            self._initialize_components()
            
            self._initialized = True
            self.logger.info("HandGestureControl initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self._safe_cleanup()
            raise

    def _initialize_logging(self):
        """Initialize logging infrastructure that will be available immediately"""
        self.logger = logging.getLogger('HandGestureControl')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _suppress_warnings(self):
        """Suppress MediaPipe and other warnings"""
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger('mediapipe').setLevel(logging.WARNING)

    def _initialize_components(self):
        """Initialize all the components of the hand gesture control"""
        # Constants and Settings
        self.wCam, self.hCam = 1280, 720
        self.frameR = 100  # Frame Reduction for mouse
        self.smoothening = 5
        self.min_click_distance = 40
        self.scroll_threshold = 25
        self.scroll_sensitivity = 1.5
        self.min_scroll_distance = 15
        self.click_hold_time = 0.3
        self.action_cooldown = 1.0
        self.SCREENSHOT_FOLDER = "screenshots"
        os.makedirs(self.SCREENSHOT_FOLDER, exist_ok=True)

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # Initialize camera
        self._initialize_camera()

        # Hand detector setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.wScr, self.hScr = pyautogui.size()

        # State variables
        self.current_mode = "mouse"  # mouse, udrl, ccpss, special
        self.last_mode_change = 0
        self.mode_cooldown = 0.5
        pyautogui.FAILSAFE = False

        # Mouse control variables
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.scroll_active = False
        self.cursor_locked = False
        self.left_click_active = False
        self.right_click_active = False
        self.last_right_click_time = 0
        self.right_click_cooldown = 0.3
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.01
        self.last_thumb_y = 0
        self.cursor_lock_pos = (0, 0)
        self.scroll_reference_y = 0
        self.click_start_time = 0
        self.is_click_hold = False

        # UDRL variables
        self.last_press_time = 0
        self.udrl_cooldown = 0.4
        self.last_fist_time = 0
        self.fist_cooldown = 0.8
        self.fist_frames = 0
        self.fist_confidence = 2
        self.prev_x, self.prev_y = None, None

        # CCPSS variables
        self.last_action_time = 0

        # Special mode variables
        self.tts_active = False
        self.last_gesture_time = 0
        self.gesture_buffer = deque(maxlen=5)

        # Finger tip IDs
        self.tipIds = [4, 8, 12, 16, 20]

    def _initialize_camera(self):
        """Initialize camera with proper error handling"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(3, self.wCam)
            self.cap.set(4, self.hCam)
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            raise

    def _safe_cleanup(self):
        """Safe cleanup that can be called during failed initialization"""
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands'):
                self.hands.close()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during safe cleanup: {str(e)}")
    def fingers_up(self, hand_landmarks):
        fingers = []
        
        # Thumb (compare x-coordinate of tip to base)
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (compare y-coordinate of tip to base)
        for id in range(1, 5):
            if hand_landmarks.landmark[self.tipIds[id]].y < hand_landmarks.landmark[self.tipIds[id]-2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

    def find_distance(self, p1, p2, landmarks, img=None, draw=True):
        x1, y1 = int(landmarks.landmark[p1].x * self.wCam), int(landmarks.landmark[p1].y * self.hCam)
        x2, y2 = int(landmarks.landmark[p2].x * self.wCam), int(landmarks.landmark[p2].y * self.hCam)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, (x1, y1, x2, y2, cx, cy)

    def is_fist(self, landmarks):
        # Check if fingertips are close to palm center
        palm_center_x = (landmarks.landmark[0].x + landmarks.landmark[9].x) / 2
        palm_center_y = (landmarks.landmark[0].y + landmarks.landmark[9].y) / 2
        
        distances = []
        for tip in [4, 8, 12, 16, 20]:  # All fingertips
            dx = landmarks.landmark[tip].x - palm_center_x
            dy = landmarks.landmark[tip].y - palm_center_y
            distances.append(math.hypot(dx, dy) * 1000)  # Scale up
        
        return all(d < 80 for d in distances)

    def handle_mouse_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Optimize by caching frequently used landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        
        # 1. Cursor Lock and Scroll Mode (thumb down)
        if fingers[0] == 1:  # Thumb down
            thumb_index_dist, _ = self.find_distance(4, 8, landmarks, img)
            
            if thumb_index_dist < self.scroll_threshold:  # SCROLL MODE
                if not self.scroll_active:
                    self.scroll_active = True
                    self.cursor_locked = True
                    self.cursor_lock_pos = (self.clocX, self.clocY)
                    self.scroll_reference_y = int(thumb_tip.y * self.hCam)
                    self.last_thumb_y = self.scroll_reference_y
                
                # Lock cursor position
                if self.cursor_lock_pos:
                    pyautogui.moveTo(*self.cursor_lock_pos)
                
                # Calculate scroll amount with deadzone and acceleration
                current_thumb_y = int(thumb_tip.y * self.hCam)
                vertical_movement = self.scroll_reference_y - current_thumb_y
                
                if abs(vertical_movement) > self.min_scroll_distance and (current_time - self.last_scroll_time) > self.scroll_cooldown:
                    # Add acceleration based on movement speed
                    scroll_factor = min(1.0, abs(vertical_movement) / (self.hCam * 0.1))  # Normalize by screen height
                    scroll_amount = int(scroll_factor * vertical_movement * self.scroll_sensitivity)
                    
                    pyautogui.scroll(scroll_amount)
                    self.last_scroll_time = current_time
                    self.scroll_reference_y = current_thumb_y
                    
                    # Only update UI if needed
                    if img is not None:
                        direction = "Scroll Up" if scroll_amount > 0 else "Scroll Down"
                        cv2.putText(img, direction, (40, 150), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            else:  # Regular cursor lock
                if not self.cursor_locked:
                    self.cursor_locked = True
                    self.cursor_lock_pos = (self.clocX, self.clocY)
                if self.cursor_lock_pos:
                    pyautogui.moveTo(*self.cursor_lock_pos)
                if img is not None:
                    cv2.putText(img, "Cursor Locked", (40, 150), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 0), 2)
                self.scroll_active = False
        else:  # Thumb up
            self.scroll_active = False
            self.cursor_locked = False
        
        # 2. Mouse Movement (index finger up, thumb up) - Optimized with prediction
        if fingers[1] == 1 and fingers[0] == 0:
            # Convert coordinates with boundary checks
            x1 = max(self.frameR, min(self.wCam - self.frameR, int(index_tip.x * self.wCam)))
            y1 = max(self.frameR, min(self.hCam - self.frameR, int(index_tip.y * self.hCam)))
            
            # Smoother interpolation with prediction
            target_x = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
            target_y = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
            
            # Predictive smoothing with velocity factor
            vel_x = (target_x - self.plocX) * 0.5  # Velocity component (adjust factor as needed)
            vel_y = (target_y - self.plocY) * 0.5
            
            self.clocX = self.plocX + (target_x - self.plocX + vel_x) / self.smoothening
            self.clocY = self.plocY + (target_y - self.plocY + vel_y) / self.smoothening
            
            # Move mouse (only if significant movement)
            if abs(self.clocX - self.plocX) > 1 or abs(self.clocY - self.plocY) > 1:
                pyautogui.moveTo(self.clocX, self.clocY)
                if img is not None:
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            self.plocX, self.plocY = self.clocX, self.clocY
        
        # 3. Left Click - Optimized with debouncing
        if fingers[0] == 0:  # Thumb up
            if fingers[1] == 1 and fingers[2] == 1:  # Index and middle up
                length, _ = self.find_distance(8, 12, landmarks, img)
                
                if length < self.min_click_distance:
                    if not self.left_click_active:
                        self.left_click_active = True
                        self.click_start_time = current_time
                        self.is_click_hold = False
                    
                    # Hold detection with visual feedback
                    if current_time - self.click_start_time > self.click_hold_time and not self.is_click_hold:
                        pyautogui.mouseDown(button='left')
                        self.is_click_hold = True
                        if img is not None:
                            cv2.putText(img, "Press-to-Select", (40, 200), 
                                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                elif self.left_click_active:
                    # Debounce release
                    if current_time - self.click_start_time > 0.05:  # 50ms debounce
                        if not self.is_click_hold:  # Single click
                            pyautogui.click(button='left')
                        else:  # Release press-to-select
                            pyautogui.mouseUp(button='left')
                        self.left_click_active = False
            elif self.left_click_active and self.is_click_hold:
                pyautogui.mouseUp(button='left')
                self.left_click_active = False
        
        # 4. Right Click - Optimized with cooldown
        if fingers[0] == 0 and not self.left_click_active:  # Thumb up and no left click active
            if fingers[4] == 1 and fingers[2] == 0:  # Pinky up, middle down
                if not self.right_click_active and (current_time - self.last_right_click_time) > self.right_click_cooldown:
                    pyautogui.click(button='right')
                    self.right_click_active = True
                    self.last_right_click_time = current_time
                    if img is not None:
                        cv2.putText(img, "Right Click!", (40, 200), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                self.right_click_active = False

    def handle_udrl_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Fist detection for ENTER
        if self.is_fist(landmarks):
            self.fist_frames += 1
            cv2.putText(img, "FIST DETECTED", (200, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            if self.fist_frames >= self.fist_confidence and current_time - self.last_fist_time > self.fist_cooldown:
                pyautogui.press("enter")
                self.last_fist_time = current_time
                self.fist_frames = 0
            return
        
        self.fist_frames = 0
        
        # Get index finger tip position
        index_tip = landmarks.landmark[8]
        current_x, current_y = int(index_tip.x * self.wCam), int(index_tip.y * self.hCam)
        
        cv2.circle(img, (current_x, current_y), 12, (0, 255, 0), -1)
        
        if self.prev_x is None:
            self.prev_x, self.prev_y = current_x, current_y
        
        delta_x = current_x - self.prev_x
        delta_y = current_y - self.prev_y
        distance = math.hypot(delta_x, delta_y)
        
        # Get direction
        direction = None
        color = (255, 255, 255)
        
        delta_y = -delta_y  # Invert y-axis
        angle = math.degrees(math.atan2(delta_y, delta_x))
        
        if -45 <= angle < 45 and distance > 60:  # Right
            direction = "right"
            color = (255, 0, 0)
        elif 45 <= angle < 135 and distance > 50:  # Up
            direction = "up"
            color = (0, 255, 0)
        elif (angle >= 135 or angle < -135) and distance > 80:  # Left
            direction = "left"
            color = (0, 0, 255)
        elif -135 <= angle < -45 and distance > 100:  # Down
            direction = "down"
            color = (0, 255, 255)
        
        if direction and distance > 30:
            cv2.line(img, (self.prev_x, self.prev_y), (current_x, current_y), color, 4)
            
            if current_time - self.last_press_time > self.udrl_cooldown:
                pyautogui.press(direction)
                self.last_press_time = current_time
        
        self.prev_x, self.prev_y = current_x, current_y

    def handle_ccpss_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Copy (thumb up only)
        if fingers == [1, 0, 0, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'c')
            self.last_action_time = current_time
            cv2.putText(img, "COPY", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        # Cut (index + middle up)
        elif fingers == [0, 1, 1, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'x')
            self.last_action_time = current_time
            cv2.putText(img, "CUT", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        # Paste (thumb + index up)
        elif fingers == [1, 1, 0, 0, 0] and current_time - self.last_action_time > self.action_cooldown:
            pyautogui.hotkey('ctrl', 'v')
            self.last_action_time = current_time
            cv2.putText(img, "PASTE", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Screenshot (all fingers up)
        elif fingers == [1, 1, 1, 1, 1] and current_time - self.last_action_time > self.action_cooldown:
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.SCREENSHOT_FOLDER, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, screenshot)
            self.last_action_time = current_time
            cv2.putText(img, "SCREENSHOT", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    def handle_special_mode(self, landmarks, img):
        fingers = self.fingers_up(landmarks)
        current_time = time.time()
        
        # Check for activation (index + middle up)
        if fingers[1] and fingers[2] and not any(fingers[3:]):
            self.gesture_buffer.append("activate")
        
        # Check for deactivation (no fingers up)
        elif not any(fingers[1:]):
            self.gesture_buffer.append("deactivate")
        else:
            self.gesture_buffer.append("none")
        
        # Check buffer for consistent gesture
        if current_time - self.last_gesture_time > self.action_cooldown:
            if self.gesture_buffer.count("activate") >= 3 and not self.tts_active:
                self.engine.say("Speech to text activated")
                self.engine.runAndWait()
                pyautogui.hotkey('win', 'h')
                self.tts_active = True
                self.last_gesture_time = current_time
            
            elif self.gesture_buffer.count("deactivate") >= 3 and self.tts_active:
                self.engine.say("Speech to text deactivated")
                self.engine.runAndWait()
                pyautogui.hotkey('win', 'h')
                self.tts_active = False
                self.last_gesture_time = current_time
        
        # Display status
        status_color = (0, 255, 0) if self.tts_active else (0, 0, 255)
        cv2.putText(img, f"TTS: {'ON' if self.tts_active else 'OFF'}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    def detect_mode(self, left_hand_landmarks):
        if not left_hand_landmarks:
            return "mouse"
        
        fingers = self.fingers_up(left_hand_landmarks)
        count = sum(fingers[1:])  # Exclude thumb
        
        if count == 1 and fingers[1] == 1:  # Index finger up
            return "udrl"
        elif count == 2 and fingers[1] == 1 and fingers[2] == 1:  # Index + middle
            return "ccpss"
        elif count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:  # Index + middle + ring
            return "special"
        return "mouse"
    def run(self):
        self.running = True
        pTime = 0
        
        try:
            while self.running:
                success, img = self.cap.read()
                if not success:
                    self.logger.warning("Failed to capture frame")
                    continue

                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                left_hand = None
                right_hand = None
                
                # Identify hands
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_type = results.multi_handedness[hand_idx].classification[0].label
                        if hand_type == "Left":
                            left_hand = hand_landmarks
                        else:
                            right_hand = hand_landmarks
                        
                        # Draw hand landmarks
                        if self.show_window:
                            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect mode from left hand
                new_mode = self.detect_mode(left_hand)
                if new_mode != self.current_mode and (time.time() - self.last_mode_change) > self.mode_cooldown:
                    self.current_mode = new_mode
                    self.last_mode_change = time.time()
                    self.logger.info(f"Mode changed to: {self.current_mode}")
                
                # Process right hand based on current mode
                if right_hand:
                    if self.current_mode == "mouse":
                        self.handle_mouse_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "udrl":
                        self.handle_udrl_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "ccpss":
                        self.handle_ccpss_mode(right_hand, img if self.show_window else None)
                    elif self.current_mode == "special":
                        self.handle_special_mode(right_hand, img if self.show_window else None)
                
                # Only show UI if window is enabled
                if self.show_window:
                    # Display mode information
                    mode_colors = {
                        "mouse": (0, 255, 255),
                        "udrl": (255, 0, 0),
                        "ccpss": (0, 255, 0),
                        "special": (255, 0, 255)
                    }
                    cv2.putText(img, f"MODE: {self.current_mode.upper()}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, mode_colors[self.current_mode], 2)
                    
                    # Display instructions
                    cv2.putText(img, "Left Hand Modes:", (10, self.hCam-120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    cv2.putText(img, "No fingers: Mouse | 1 finger: UDRL", (10, self.hCam-90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    cv2.putText(img, "2 fingers: CCPSS | 3 fingers: Special", (10, self.hCam-60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                    
                    # Display FPS
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(img, f"FPS: {int(fps)}", (self.wCam-150, 50), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                    cv2.imshow("Integrated Hand Control", img)
                
                # Check for quit command (even if window is hidden)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

        except Exception as e:
            self.logger.error(f"Error in hand gesture control: {str(e)}")
        finally:
            self.stop()

    def start(self):
        """Start the hand gesture control in a separate thread"""
        if not self.running:
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Hand gesture control started")

    def stop(self):
        """Stop the hand gesture control and clean up resources"""
        if not hasattr(self, '_initialized') or not self._initialized:
            return

        self.running = False
        
        try:
            if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'show_window') and self.show_window:
                cv2.destroyAllWindows()
            self.logger.info("Hand gesture control stopped")
        except Exception as e:
            self.logger.error(f"Error during stop: {str(e)}")

    def __del__(self):
        """Destructor to ensure resources are released"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info("Cleaning up HandGestureControl resources")
            self.stop()
        except Exception:
            pass