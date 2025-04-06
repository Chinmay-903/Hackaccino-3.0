import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
from datetime import datetime
import time
from PyQt5.QtWidgets import QApplication, QFileDialog

class FaceStateManager:
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        self.face_detected = False
        self.recognition_done = False
        self.liveness_verified = False
        self.no_face_frames = 0
        self.face_verified = False
        self.recognized_name = None
        self.countdown_start = None
        self.countdown_seconds = 5
        self.setup_mode = False
        self.setup_complete = False

class LivenessDetector:
    def __init__(self):
        self.EAR_THRESHOLD = 0.21
        self.REQUIRED_BLINKS = 2
        self.eye_landmarks = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144]
        }
        self.ear_history = []
        self.blink_count = 0
        self.blink_cooldown = 0
        self.eye_open = True

    def reset(self):
        self.ear_history = []
        self.blink_count = 0
        self.blink_cooldown = 0
        self.eye_open = True

    def calculate_ear(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear

    def update(self, face_landmarks, img_shape):
        h, w = img_shape[:2]
        ear_values = []
        
        for side in ['left', 'right']:
            eye_points = []
            for idx in self.eye_landmarks[side]:
                lm = face_landmarks.landmark[idx]
                eye_points.append((lm.x * w, lm.y * h))
            ear = self.calculate_ear(np.array(eye_points))
            ear_values.append(ear)
        
        avg_ear = sum(ear_values) / 2.0
        self.ear_history.append(avg_ear)
        
        if len(self.ear_history) > 5:
            self.ear_history.pop(0)

        if avg_ear < self.EAR_THRESHOLD and self.eye_open:
            self.eye_open = False
            self.blink_count += 1
            self.blink_cooldown = 10
        elif avg_ear > self.EAR_THRESHOLD and not self.eye_open:
            self.eye_open = True

        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1

        return self.blink_count >= self.REQUIRED_BLINKS

class FaceAuthSystem:
    def __init__(self, known_faces_path='photos', recognition_threshold=0.55):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.state = FaceStateManager()
        self.liveness_detector = LivenessDetector()
        self.cyber_blue = (255, 191, 0)
        self.key_landmarks = {
            1: ("NOSE", "left"),
            33: ("LEFT EYE", "left"),
            263: ("RIGHT EYE", "right"),
            61: ("MOUTH LEFT", "left"),
            291: ("MOUTH RIGHT", "right"),
            199: ("CHIN", "left")
        }
        self.recognition_threshold = recognition_threshold
        self.known_faces_path = known_faces_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.setup_known_faces()

    def setup_known_faces(self):
        if not os.path.exists(self.known_faces_path):
            os.makedirs(self.known_faces_path, exist_ok=True)
            self.state.setup_mode = True
            return

        image_files = [f for f in os.listdir(self.known_faces_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in image_files:
            image_path = os.path.join(self.known_faces_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if face_locations:
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        
        if not self.known_face_encodings and not self.state.setup_complete:
            img = self.show_setup_prompt(img)
            return img
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_img = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_img)
        
        if face_locations:
            results = self.face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                self.handle_detection(img, results)
                for face_landmarks in results.multi_face_landmarks:
                    self.draw_landmarks(img, face_landmarks)
            else:
                self.handle_no_detection()
        else:
            self.handle_no_detection()

        self.draw_recognition_result(img)
        return img

    def show_setup_prompt(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.putText(img, "FACE ID SETUP REQUIRED", (w//2 - 200, h//2 - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.cyber_blue, 2)
        cv2.putText(img, "1. Look straight at the camera", (w//2 - 200, h//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "2. Blink twice when prompted", (w//2 - 200, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "OR Press 'S' to upload an image", (w//2 - 200, h//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def handle_image_upload(self):
        app = QApplication.instance() or QApplication([])
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                uploaded_img = cv2.imread(image_path)
                if uploaded_img is not None:
                    return self.process_uploaded_image(uploaded_img)
        return False

    def process_uploaded_image(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            print("Error: No face detected in the uploaded image")
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_uploaded_{timestamp}.jpg"
        save_path = os.path.join(self.known_faces_path, filename)
        cv2.imwrite(save_path, img)
        print(f"Successfully saved face image to: {save_path}")
        
        encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(os.path.splitext(filename)[0])
        self.state.setup_complete = True
        self.state.setup_mode = False
        return True

    def handle_detection(self, img, results):
        self.state.no_face_frames = 0
        self.state.face_detected = True

        for face_landmarks in results.multi_face_landmarks:
            liveness_verified = self.liveness_detector.update(face_landmarks, img.shape)
            
            if liveness_verified:
                self.state.liveness_verified = True
                if not self.state.recognition_done:
                    if self.state.setup_mode:
                        self.register_new_face(img)
                    else:
                        recognized = self.run_recognition(img)
                        if recognized:
                            self.state.face_verified = True
                            if self.state.countdown_start is None:
                                self.state.countdown_start = time.time()
                        self.state.recognition_done = True

    def register_new_face(self, img):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_{timestamp}.jpg"
        cv2.imwrite(os.path.join(self.known_faces_path, filename), img)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if face_locations:
            encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(os.path.splitext(filename)[0])
            self.state.setup_complete = True
            self.state.setup_mode = False

    def handle_no_detection(self):
        self.state.no_face_frames += 1
        if self.state.no_face_frames > 15:
            self.state.reset_state()
            self.liveness_detector.reset()

    def run_recognition(self, img):
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_img)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)
            if face_encodings:
                return self.recognize_face(face_encodings[0])
        return False

    def recognize_face(self, encode_face):
        if not self.known_face_encodings:
            return False
            
        face_distances = face_recognition.face_distance(self.known_face_encodings, encode_face)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] < self.recognition_threshold:
            self.state.recognized_name = self.known_face_names[best_match_index].upper()
            self.mark_attendance(self.state.recognized_name)
            return True
        return False

    def mark_attendance(self, name, output_file='Attendance.csv'):
        with open(output_file, 'a') as f:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dt_string}')

    def draw_landmarks(self, img, face_landmarks):
        h, w, c = img.shape
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in self.key_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                feature_name, position = self.key_landmarks[idx]
                
                if self.state.face_verified:
                    status = "VERIFIED"
                    text_color = (0, 255, 0)
                elif self.state.recognized_name is None and self.state.liveness_verified:
                    status = "UNKNOWN FACE"
                    text_color = (0, 255, 255)
                else:
                    status = "NOT VERIFIED"
                    text_color = (0, 0, 255)
                
                self.draw_feature_annotation(img, x, y, feature_name, position, status, text_color)

    def draw_feature_annotation(self, frame, x, y, feature, position, status, color):
        line_color = self.cyber_blue
        text = f"{feature}\n{status}"
        font_scale = 0.4
        thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        box_padding = 10
        box_width = text_width + box_padding * 2
        box_height = text_height + box_padding * 2 + 10
        
        if position == "left":
            box_x = x - box_width - 50
            line_start = (x - 5, y)
        else:
            box_x = x + 50
            line_start = (x + 5, y)
            
        box_y = y - box_height // 2
        line_end = (box_x + box_width if position == "left" else box_x, box_y + box_height // 2)

        box_x = max(10, min(box_x, frame.shape[1] - box_width - 10))
        box_y = max(10, min(box_y, frame.shape[0] - box_height - 10))

        cv2.line(frame, line_start, line_end, line_color, 1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), line_color, 1)
        
        text_x = box_x + box_padding
        text_y = box_y + box_padding + 15
        for i, line in enumerate(text.split('\n')):
            cv2.putText(frame, line, (text_x, text_y + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        if not self.state.liveness_verified and feature in ["LEFT EYE", "RIGHT EYE"]:
            cv2.putText(frame, "Blink to verify", (x, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def draw_recognition_result(self, img):
        if self.state.face_verified:
            if self.state.countdown_start is not None:
                elapsed = time.time() - self.state.countdown_start
                remaining = max(0, self.state.countdown_seconds - int(elapsed))
                status_text = f"FACE VERIFIED: Launching in {remaining}s"
            else:
                status_text = f"FACE VERIFIED: {self.state.recognized_name}"
            color = (0, 255, 0)
        elif self.state.face_detected:
            status_text = "FACE NOT VERIFIED"
            color = (0, 0, 255)
        else:
            status_text = "Liveness: Checking..."
            color = (0, 255, 255)
        
        cv2.putText(img, "SYSTEM STATUS", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cyber_blue, 1)
        cv2.putText(img, status_text, (40, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def should_launch_app(self):
        if self.state.countdown_start is None:
            return False
        elapsed = time.time() - self.state.countdown_start
        return elapsed >= self.state.countdown_seconds

    def get_system_state(self):
        return {
            'face_detected': self.state.face_detected,
            'face_verified': self.state.face_verified,
            'recognized_name': self.state.recognized_name,
            'liveness_verified': self.state.liveness_verified,
            'countdown_complete': self.should_launch_app(),
            'setup_mode': self.state.setup_mode,
            'setup_complete': self.state.setup_complete
        }