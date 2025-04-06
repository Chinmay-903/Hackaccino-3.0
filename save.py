import sys
import PyInstaller
PyInstaller.__main__.PYI_DISABLE_BYTECODE = True  # Disable bytecode analysis

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress absl warnings

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import pyautogui
import requests
import io
import threading
import time

# Configure Google Generative AI
genai.configure(api_key="Your api key")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam with optimal settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize hand detector
detector = HandDetector(
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.8,
    minTrackCon=0.7
)

# Configuration
response_cache = {}
IMAGE_CACHE_SIZE = 5
RECOGNITION_COOLDOWN = 3
IMAGE_DISPLAY_TIME = 5  # Seconds to display image before returning to camera

# State variables
prev_pos = None
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
last_recognition = ""
last_recognition_time = 0
display_mode = "camera"  # "camera" or "image"
current_image = None
image_display_start = 0


def getHandInfo(img):
    try:
        hands, img = detector.findHands(img, draw=False, flipType=True)
        if not hands:
            return None
        hand = hands[0]
        return detector.fingersUp(hand), hand["lmList"]
    except Exception as e:
        print(f"Hand detection error: {e}")
        return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Drawing mode
        current_pos = lmList[8][0:2]
        if prev_pos:
            cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, current_pos, 5, (255, 0, 255), -1)
    elif fingers == [1, 0, 0, 0, 0]:  # Clear canvas
        canvas.fill(0)
        global display_mode
        display_mode = "camera"  # Return to camera view when clearing

    return current_pos, canvas


def recognize_drawing(model, canvas):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        response = model.generate_content([
            """You are an expert at identifying hand-drawn sketches. Analyze the drawing and:
            1. Identify the SINGLE most prominent object being depicted
            2. Return ONLY the most specific, simple noun that describes it
            3. Use lowercase with no punctuation or extra words
            4. If unclear or abstract, return 'unknown'

            Important rules:
            - Be literal (interpret drawings at face value)
            - Prefer concrete objects over abstract concepts
            - Choose the most complete interpretation
            - Ignore minor imperfections in the drawing

            Examples of good responses:
            - 'cat' (not 'animal')
            - 'car' (not 'vehicle')
            - 'tree' (not 'plant')
            - 'house' (not 'building')
            - 'unknown' (if unclear)""",
            pil_image
        ], request_options={"timeout": 3})
        return response.text.strip().lower()
    except Exception as e:
        print(f"AI recognition error: {e}")
        return None


def fetch_stock_image_async(query, callback):
    def fetch():
        if query in response_cache:
            callback(response_cache[query])
            return

        try:
            api_key = 'your api key'
            url = f'https://api.unsplash.com/photos/random?query={query}&client_id={api_key}'
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                image_url = data['urls']['regular']
                img_response = requests.get(image_url, timeout=5)
                image = Image.open(io.BytesIO(img_response.content))

                if len(response_cache) >= IMAGE_CACHE_SIZE:
                    response_cache.pop(next(iter(response_cache)))
                response_cache[query] = image

                callback(image)
        except Exception as e:
            print(f"Image fetch error: {e}")

    threading.Thread(target=fetch, daemon=True).start()


def process_image_display():
    global display_mode, image_display_start
    display_mode = "image"
    image_display_start = time.time()


try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        info = getHandInfo(img)

        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)

            # Recognition trigger with cooldown
            current_time = time.time()
            if (fingers == [0, 1, 1, 1, 1] and
                    current_time - last_recognition_time > RECOGNITION_COOLDOWN):
                recognition = recognize_drawing(model, canvas)
                if recognition and recognition != "unknown" and recognition != last_recognition:
                    print(f"Detected: {recognition}")
                    pyautogui.write(recognition)
                    last_recognition = recognition
                    last_recognition_time = current_time


                    def update_canvas(image):
                        global current_image
                        current_image = cv2.resize(np.array(image), (canvas.shape[1], canvas.shape[0]))
                        process_image_display()


                    fetch_stock_image_async(recognition, update_canvas)

        # Check if we should return to camera view
        if display_mode == "image" and (time.time() - image_display_start) > IMAGE_DISPLAY_TIME:
            display_mode = "camera"

        # Display the appropriate view
        if display_mode == "image" and current_image is not None:
            # Show just the canvas with the fetched image
            display_img = current_image.copy()
        else:
            # Show camera feed with drawing overlay
            display_img = img.copy()
            cv2.addWeighted(display_img, 0.7, canvas, 0.3, 0, dst=display_img)

        # Show FPS and mode indicator
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(display_img, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, f"Mode: {display_mode}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking and Drawing", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram gracefully interrupted by user")

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    # Ensure proper cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program resources released properly")
