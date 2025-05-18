import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist # For EAR calculation

# Constants for detection
# These values might need fine-tuning
EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR = 0.15  # Factor for eyebrow-to-eye distance increase for "No"
CALIBRATION_FRAMES = 60  # Number of frames for initial calibration
EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold for eye closure (Yes)

# Path to the dlib shape predictor model file
DLIB_SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Display states
STATE_YES = "Yes"
STATE_NO = "No"
STATE_NORMAL = "Normal"
STATE_CALIBRATING = "Calibrating..."

# Initialize dlib's face detector and facial landmark predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_PATH)
except RuntimeError as e:
    print(f"[ERROR] Failed to load dlib model: {e}")
    print(f"Please download '{DLIB_SHAPE_PREDICTOR_PATH}' and place it near the script.")
    exit()

# Landmark indices for eyes (used for EAR)
# Dlib's left_eye (user's left) are points 42-47 (0-indexed in predictor)
# Dlib's right_eye (user's right) are points 36-41
(user_L_eye_indices_start, user_L_eye_indices_end) = (42, 48)
(user_R_eye_indices_start, user_R_eye_indices_end) = (36, 42)

# Landmark indices for top of eyes (for No detection)
# User's Left eye top: dlib points 43, 44
# User's Right eye top: dlib points 37, 38
user_L_eye_top_indices = [43, 44]
user_R_eye_top_indices = [37, 38]

# Landmark indices for eyebrows (for No detection)
# User's Left eyebrow: dlib points 22-26. We'll average points 23, 24, 25.
user_L_eyebrow_y_calc_indices = range(23, 26) 
# User's Right eyebrow: dlib points 17-21. We'll average points 18, 19, 20.
user_R_eyebrow_y_calc_indices = range(18, 21)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

# Calibration variables
calibration_counter = 0
# For eyebrows
normal_user_L_eyebrow_y_avg = 0
normal_user_R_eyebrow_y_avg = 0
calibration_data_user_L_eyebrow_y = []
calibration_data_user_R_eyebrow_y = []
# For top of eyes
normal_user_L_eye_top_y_avg = 0
normal_user_R_eye_top_y_avg = 0
calibration_data_user_L_eye_top_y = []
calibration_data_user_R_eye_top_y = []
# For calibrated distances
normal_dist_L_eyebrow_to_eye = 0
normal_dist_R_eyebrow_to_eye = 0

def get_landmark_point(landmarks, index):
    return (landmarks.part(index).x, landmarks.part(index).y)

def eye_aspect_ratio(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    ear_val = (A + B) / (2.0 * C)
    return ear_val

print("[INFO] Calibration started. Please look at the camera with a normal expression...")
current_state = STATE_CALIBRATING

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if calibration_counter >= CALIBRATION_FRAMES:
        current_state = STATE_NORMAL

    for face in faces:
        landmarks = predictor(gray, face)

        # --- Calculate current eyebrow Y positions ---
        user_L_eyebrow_current_y_pts = [landmarks.part(i).y for i in user_L_eyebrow_y_calc_indices]
        current_user_L_eyebrow_y_avg = np.mean(user_L_eyebrow_current_y_pts) if user_L_eyebrow_current_y_pts else 0

        user_R_eyebrow_current_y_pts = [landmarks.part(i).y for i in user_R_eyebrow_y_calc_indices]
        current_user_R_eyebrow_y_avg = np.mean(user_R_eyebrow_current_y_pts) if user_R_eyebrow_current_y_pts else 0

        # --- Calculate current top of eye Y positions ---
        user_L_eye_top_current_y_pts = [landmarks.part(i).y for i in user_L_eye_top_indices]
        current_user_L_eye_top_y_avg = np.mean(user_L_eye_top_current_y_pts) if user_L_eye_top_current_y_pts else 0
        
        user_R_eye_top_current_y_pts = [landmarks.part(i).y for i in user_R_eye_top_indices]
        current_user_R_eye_top_y_avg = np.mean(user_R_eye_top_current_y_pts) if user_R_eye_top_current_y_pts else 0

        # --- Eye Aspect Ratio for "Yes" detection ---
        user_L_eye_all_pts = np.array([get_landmark_point(landmarks, i) for i in range(user_L_eye_indices_start, user_L_eye_indices_end)], dtype="int")
        user_R_eye_all_pts = np.array([get_landmark_point(landmarks, i) for i in range(user_R_eye_indices_start, user_R_eye_indices_end)], dtype="int")
        
        left_ear = eye_aspect_ratio(user_L_eye_all_pts)
        right_ear = eye_aspect_ratio(user_R_eye_all_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # --- Calibration Phase ---
        if calibration_counter < CALIBRATION_FRAMES:
            current_state = STATE_CALIBRATING
            calibration_data_user_L_eyebrow_y.append(current_user_L_eyebrow_y_avg)
            calibration_data_user_R_eyebrow_y.append(current_user_R_eyebrow_y_avg)
            calibration_data_user_L_eye_top_y.append(current_user_L_eye_top_y_avg)
            calibration_data_user_R_eye_top_y.append(current_user_R_eye_top_y_avg)
            calibration_counter += 1
            
            if calibration_counter == CALIBRATION_FRAMES:
                normal_user_L_eyebrow_y_avg = np.mean(calibration_data_user_L_eyebrow_y)
                normal_user_R_eyebrow_y_avg = np.mean(calibration_data_user_R_eyebrow_y)
                normal_user_L_eye_top_y_avg = np.mean(calibration_data_user_L_eye_top_y)
                normal_user_R_eye_top_y_avg = np.mean(calibration_data_user_R_eye_top_y)

                # Distance = Y_eye_top - Y_eyebrow (larger means eyebrow is higher or eye is lower)
                normal_dist_L_eyebrow_to_eye = normal_user_L_eye_top_y_avg - normal_user_L_eyebrow_y_avg
                normal_dist_R_eyebrow_to_eye = normal_user_R_eye_top_y_avg - normal_user_R_eyebrow_y_avg
                
                print("[INFO] Calibration finished.")
                print(f"[INFO] Calibrated Avg L Eyebrow Y: {normal_user_L_eyebrow_y_avg:.2f}, Avg R Eyebrow Y: {normal_user_R_eyebrow_y_avg:.2f}")
                print(f"[INFO] Calibrated Avg L Eye Top Y: {normal_user_L_eye_top_y_avg:.2f}, Avg R Eye Top Y: {normal_user_R_eye_top_y_avg:.2f}")
                print(f"[INFO] Calibrated Dist L Eyebrow-Eye: {normal_dist_L_eyebrow_to_eye:.2f}, R Eyebrow-Eye: {normal_dist_R_eyebrow_to_eye:.2f}")
                print(f"[INFO] 'No' detection if current distance > normal_dist * (1 + {EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR})")
                print(f"[INFO] 'Yes' detection threshold (EAR must be <): {EAR_THRESHOLD:.2f}")

        # --- Detection Phase ---
        else:
            if normal_dist_L_eyebrow_to_eye != 0 and normal_dist_R_eyebrow_to_eye != 0: # Ensure calibration is done
                
                # Detect "Yes" (eyes closed)
                if avg_ear < EAR_THRESHOLD:
                    current_state = STATE_YES
                
                # Detect "No" (eyebrows raised significantly relative to eyes)
                else: # Check for NO only if not YES
                    current_dist_L = current_user_L_eye_top_y_avg - current_user_L_eyebrow_y_avg
                    current_dist_R = current_user_R_eye_top_y_avg - current_user_R_eyebrow_y_avg

                    threshold_dist_L = normal_dist_L_eyebrow_to_eye * (1 + EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR)
                    threshold_dist_R = normal_dist_R_eyebrow_to_eye * (1 + EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR)
                    
                    # Handle cases where normal distance might be zero or negative (eyebrow very low)
                    # If normal distance is small or negative, a small absolute increase might be enough.
                    # This part might need more sophisticated handling if eyebrows are naturally very low or touching eyelids.
                    # For now, simple multiplication factor.
                    if normal_dist_L_eyebrow_to_eye <= 0: threshold_dist_L = normal_dist_L_eyebrow_to_eye + abs(normal_dist_L_eyebrow_to_eye * EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR) + 5 # Add a small fixed increase
                    if normal_dist_R_eyebrow_to_eye <= 0: threshold_dist_R = normal_dist_R_eyebrow_to_eye + abs(normal_dist_R_eyebrow_to_eye * EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR) + 5


                    if current_dist_L > threshold_dist_L and current_dist_R > threshold_dist_R:
                        current_state = STATE_NO
        
        # Optional: Draw landmarks for debugging
        # Eyebrows
        # for i in list(user_L_eyebrow_y_calc_indices) + list(user_R_eyebrow_y_calc_indices):
        #     p = get_landmark_point(landmarks, i)
        #     cv2.circle(frame, p, 2, (0, 255, 0), -1)
        # # Top of eyes
        # for i in user_L_eye_top_indices + user_R_eye_top_indices:
        #     p = get_landmark_point(landmarks, i)
        #     cv2.circle(frame, p, 2, (255, 0, 0), -1)

    # Display the detected state
    display_text = current_state
    color = (255, 255, 0) # Default for Normal/Calibrating
    if current_state == STATE_YES:
        color = (0, 255, 0) # Green for Yes
    elif current_state == STATE_NO:
        color = (0, 0, 255) # Red for No

    cv2.putText(frame, display_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Gesture Detection (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Application closed.") 