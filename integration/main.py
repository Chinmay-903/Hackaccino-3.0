import sys
import cv2
import mediapipe as mp
import time
from PyQt5.QtWidgets import QApplication
from face_system.face_auth import FaceAuthSystem
from launcher.app_launcher import ModernSoftwareLauncher
from HandGestureControl.hand_gesture_control import HandGestureControl  # Import the hand gesture module

class AppController:
    def __init__(self):
        self.face_system = FaceAuthSystem()
        self.last_validation_time = 0
        self.validation_interval = 10000000000000000000000
        self.cap = None
        self.hand_gesture = None  # Will hold our hand gesture controller

    def run_initial_validation(self):
        self.cap = cv2.VideoCapture(0)
        pTime = 0

        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = self.face_system.process_frame(img)
            
            # Show FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (img.shape[1] - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Face Verification", img)
            
            key = cv2.waitKey(1)
            if key == ord('s') or key == ord('S'):
                cv2.destroyAllWindows()
                if self.cap:
                    self.cap.release()
                
                # Handle image upload
                if self.face_system.handle_image_upload():
                    # After successful upload, restart face unlock
                    self.cap = cv2.VideoCapture(0)
                    self.face_system.state.reset_state()
                    continue
                else:
                    # If upload failed, restart face unlock
                    self.cap = cv2.VideoCapture(0)
                    continue
            
            state = self.face_system.get_system_state()
            
            if state['countdown_complete']:
                if self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
                self.last_validation_time = time.time()
                
                # Initialize hand gesture control before launching app
                self.hand_gesture = HandGestureControl()
                self.hand_gesture.start()
                
                self.launch_app()
                break
            
            if key == ord('q'):
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def launch_app(self):
        app = QApplication.instance() or QApplication([])
        self.launcher = ModernSoftwareLauncher(self.validate_user)
        self.launcher.show()
        
        # Connect the launcher's close event to stop the hand gesture control
        self.launcher.destroyed.connect(self.cleanup)
        
        app.exec_()

    def cleanup(self):
        """Clean up resources when the app is closing"""
        if self.hand_gesture:
            self.hand_gesture.stop()
            self.hand_gesture = None

    def validate_user(self):
        current_time = time.time()
        if current_time - self.last_validation_time > self.validation_interval:
            # Temporarily stop hand gesture control during validation
            if self.hand_gesture:
                self.hand_gesture.stop()
            
            cap = cv2.VideoCapture(0)
            start_time = time.time()
            validated = False
            
            while time.time() - start_time < 3:
                success, img = cap.read()
                if not success:
                    continue
                
                img = self.face_system.process_frame(img)
                state = self.face_system.get_system_state()
                
                if state['countdown_complete']:
                    validated = True
                    self.last_validation_time = time.time()
                    break
            
            cap.release()
            
            # Restart hand gesture control after validation
            if validated and not self.hand_gesture:
                self.hand_gesture = HandGestureControl()
                self.hand_gesture.start()
            
            return validated
        return True

def main():
    controller = AppController()
    controller.run_initial_validation()

if __name__ == "__main__":
    main()