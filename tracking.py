import cv2
import numpy as np
import time
import threading

BOARD_SIZE = (10, 7)
DETECTION_SCALE = 0.4
ROI_MARGIN = 50

# Use GStreamer pipeline for libcamera
gst_str = (
    "libcamerasrc ! "
    "video/x-raw,width=1280,height=720,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

cap.set(cv2.CAP_PROP_FPS, 60)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

last_corners = None
corners_lock = threading.Lock()
found_flag = False

fps_buffer = []
delay_buffer = []
FPS_BUFFER_SIZE = 10
DELAY_BUFFER_SIZE = 10

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    debug_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = False
    corners = None

    if last_corners is not None:
        x, y, w, h = cv2.boundingRect(last_corners.astype(np.int32))
        x0 = max(x - ROI_MARGIN, 0)
        y0 = max(y - ROI_MARGIN, 0)
        x1 = min(x + w + ROI_MARGIN, gray.shape[1])
        y1 = min(y + h + ROI_MARGIN, gray.shape[0])
        roi = gray[y0:y1, x0:x1]
        roi_resized = cv2.resize(roi, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        found, corners = cv2.findChessboardCorners(roi_resized, BOARD_SIZE, None)
        if found:
            corners = corners / DETECTION_SCALE
            corners[:, 0, 0] += x0
            corners[:, 0, 1] += y0
    else:
        gray_resized = cv2.resize(gray, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        found, corners = cv2.findChessboardCorners(gray_resized, BOARD_SIZE, None)
        if found:
            corners = corners / DETECTION_SCALE

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        last_corners = corners2
        found_flag = True
    else:
        last_corners = None
        found_flag = False

    if last_corners is not None and found_flag:
        corners2 = last_corners.copy()
    else:
        corners2 = None

    if corners2 is not None:
        cv2.drawChessboardCorners(debug_frame, BOARD_SIZE, corners2, True)
        top_left = corners2[0][0]
        dx = (corners2[BOARD_SIZE[0] - 1][0][0] - top_left[0]) / (BOARD_SIZE[0] - 1)
        dy = (corners2[-1][0][1] - top_left[1]) / (BOARD_SIZE[1] - 1)

        for i in range(BOARD_SIZE[0] + 1):
            x = int(top_left[0] + i * dx)
            y1 = int(top_left[1])
            y2 = int(top_left[1] + BOARD_SIZE[1] * dy)
            cv2.line(debug_frame, (x, y1), (x, y2), (0, 255, 0), 1)

        for i in range(BOARD_SIZE[1] + 1):
            y = int(top_left[1] + i * dy)
            x1 = int(top_left[0])
            x2 = int(top_left[0] + BOARD_SIZE[0] * dx)
            cv2.line(debug_frame, (x1, y), (x2, y), (0, 255, 0), 1)
    frame_time = time.time() - start_time
    fps = 1.0 / frame_time if frame_time > 0 else 0
    delay_ms = int(frame_time * 1000)

    # Update FPS buffer
    fps_buffer.append(fps)
    if len(fps_buffer) > FPS_BUFFER_SIZE:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    # Update delay buffer
    delay_buffer.append(delay_ms)
    if len(delay_buffer) > DELAY_BUFFER_SIZE:
        delay_buffer.pop(0)
    avg_delay = sum(delay_buffer) / len(delay_buffer)

    cv2.putText(debug_frame, f"Avg FPS: {avg_fps:.2f} | Avg Delay: {avg_delay:.0f} ms",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)
    cv2.putText(debug_frame, f"Camera FPS: {cam_fps:.2f}",
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

    cv2.imshow("Chess Detection", debug_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(cv2.getBuildInformation())
