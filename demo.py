import cv2
import numpy as np
import time

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

prev_gray_face = None
prev_face_coords = None
blink_frame_count = 0
blink_required_frames = 2
last_blink_time = 0
last_motion_time = 0
motion_threshold = 5.0
liveness_check_duration = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    label = "Fake"
    color = (0, 0, 255)  # red by default

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)

        # Blink detection
        is_blinking = len(eyes) < 2
        if is_blinking:
            blink_frame_count += 1
        else:
            if blink_frame_count >= blink_required_frames:
                last_blink_time = time.time()
            blink_frame_count = 0

        # Local motion detection in eye region
        current_face = cv2.GaussianBlur(face_gray, (5, 5), 0)
        if prev_gray_face is not None and current_face.shape == prev_gray_face.shape:
            diff = cv2.absdiff(prev_gray_face, current_face)
            motion_value = np.mean(diff)
            if motion_value > motion_threshold:
                last_motion_time = time.time()

        prev_gray_face = current_face

        # Optional: Check face position consistency
        if prev_face_coords:
            px, py, pw, ph = prev_face_coords
            dx = abs(x - px)
            dy = abs(y - py)
            dw = abs(w - pw)
            dh = abs(h - ph)
            if dx + dy + dw + dh > 20:  # Significant face position/size change
                last_motion_time = time.time()

        prev_face_coords = (x, y, w, h)

        # Liveness decision
        current_time = time.time()
        if (current_time - last_blink_time) < liveness_check_duration and \
           (current_time - last_motion_time) < liveness_check_duration:
            label = "Real"
            color = (0, 255, 0)
        else:
            label = "Fake"
            color = (0, 0, 255)

        # Draw result
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Liveness Detection - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()