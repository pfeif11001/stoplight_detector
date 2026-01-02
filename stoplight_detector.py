import cv2
import platform
import numpy as np
import sys
import os

# --- CROSS-PLATFORM BEEP ---
def beep():
    if sys.platform.startswith("win"):
        import winsound
        winsound.Beep(1000, 200)  # 1000Hz, 200ms
    else:
        # Pi/Linux: play beep.wav in the same directory
        os.system("aplay -q beep.wav")

# --- CAMERA SETUP ---
CAMERA_INDEX = 0  # your camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 640, 480)

print("Camera started. Press ESC or Ctrl+C to exit.")

# --- INITIALIZE LAST KNOWN CIRCLE ---
last_center = None
last_radius = None
last_color = (0, 0, 255)  # default red
prev_color_name = "RED"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected or frame grab failed.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_score = 0
        best_center = None
        best_radius = None
        best_color = None
        color_name = None

        # ---------------- RED DETECTION ----------------
        mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))
        red_mask = cv2.medianBlur(mask1 | mask2, 5)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 5000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = max(4 * np.pi * (area / (perimeter**2)), 0.1)
            mask = np.zeros(red_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            avg_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0] / 255.0
            avg_val = cv2.mean(hsv[:, :, 2], mask=mask)[0] / 255.0
            score = area * (avg_sat**1.5) * (avg_val**1.5) * circularity
            if score > best_score:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                best_score = score
                best_center = (int(x), int(y))
                best_radius = int(radius)
                best_color = (0, 0, 255)
                color_name = "RED"

        # ---------------- GREEN DETECTION ----------------
        green_mask = cv2.inRange(hsv, (35, 50, 50), (100, 255, 255))
        green_mask = cv2.medianBlur(green_mask, 5)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 5000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = max(4 * np.pi * (area / (perimeter**2)), 0.1)
            mask = np.zeros(green_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            avg_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0] / 255.0
            avg_val = cv2.mean(hsv[:, :, 2], mask=mask)[0] / 255.0
            score = area * (avg_sat**1.5) * (avg_val**1.5) * circularity
            if score > best_score:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                best_score = score
                best_center = (int(x), int(y))
                best_radius = int(radius)
                best_color = (0, 255, 0)
                color_name = "GREEN"

        # Update last known circle
        if best_center is not None and best_radius is not None:
            last_center = best_center
            last_radius = best_radius
            last_color = best_color

        # Draw sticky circle
        if last_center is not None and last_radius is not None:
            cv2.circle(frame, last_center, last_radius, last_color, 3)

        # Play beep on RED -> GREEN
        if prev_color_name == "RED" and color_name == "GREEN":
            beep()
        prev_color_name = color_name if color_name is not None else prev_color_name

        cv2.imshow("Camera", frame)

        # ESC key exits
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

except KeyboardInterrupt:
    print("\nExiting via Ctrl+C")

cap.release()
cv2.destroyAllWindows()
print("Camera closed cleanly.")
