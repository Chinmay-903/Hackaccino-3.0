import os
import cv2
import face_recognition
from datetime import datetime

def save_uploaded_image(image, output_dir='photos'):
    """Save uploaded image and return its encoding"""
    os.makedirs(output_dir, exist_ok=True)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return None, "No face detected in the uploaded image"
    
    encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user_uploaded_{timestamp}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), image)
    
    return encoding, filename