import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame
import time
import os

# --- FIX FOR THE 'SDL' / 'OBJC' CONFLICT ---
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --- STEP 1: INITIALIZE AUDIO ---
pygame.mixer.init()
try:
    # Load your sound file
    alarm_sound = pygame.mixer.Sound("/Users/ayushkhandelwal/Downloads/beep-06.wav")
except:
    print("[WARNING] alarm.wav not found in folder. Sound will not play.")
    alarm_sound = None

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- STEP 2: THRESHOLDS ---
EYE_AR_THRESH = 0.22      # Adjust this based on your on-screen EAR
EYE_AR_CONSEC_FRAMES = 40 # ~2 seconds of closure
COUNTER = 0
ALARM_ON = False

# --- STEP 3: MODELS ---
detector = dlib.get_frontal_face_detector()
# Using the path you provided earlier
PATH_TO_DAT = "/Users/ayushkhandelwal/Downloads/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PATH_TO_DAT)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

        # Draw visual markers
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # --- STEP 4: STABLE JUDGMENT LOGIC ---
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # Trigger Alarm
                if not ALARM_ON:
                    ALARM_ON = True
                    if alarm_sound:
                        alarm_sound.play(-1) # -1 means loop forever
                
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset Alarm
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                if alarm_sound:
                    alarm_sound.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
if alarm_sound:
    alarm_sound.stop()
cap.release()
cv2.destroyAllWindows()