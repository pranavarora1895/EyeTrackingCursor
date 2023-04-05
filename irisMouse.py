"""
PROJECT: EYE TRACKING CURSOR
COURSE: COMP6982 COMPUTER VISION
GROUP MEMBERS:
 1. RABIA WAHEED: 202193590 (rwaheed@mun.ca)
 2. PRANAV ARORA: 202286040 (parora@mun.ca)
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pyautogui
from enum import Enum

mp_face_mesh = mp.solutions.face_mesh
cap = cv.VideoCapture(0)

screen_w, screen_h = pyautogui.size()


class EYE_MESH_LANDMARKS(Enum):
    """MediaPipe specific iris mesh landmarks"""
    RIGHT_IRIS = [474, 475, 476, 477]
    LEFT_IRIS = [469, 470, 471, 472]
    L_H_LEFT = [33]  # right eye's right most landmark
    L_H_RIGHT = [133]  # right eye's left most landmark
    R_H_LEFT = [362]  # left eye right most landmark
    R_H_RIGHT = [263]  # left eye left most landmark
    LEFT_EYE_CLOSE_THRESHOLD = 1.5


def euclidean_distance(pointA, pointB):
    """Calculates the euclidean distance from iris to the end points of the eye"""
    x1, y1 = pointA.ravel()
    x2, y2 = pointB.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def iris_position(iris_center, right_point, left_point):
    """Determines the iris position whether it's in the left, center or right.
    Also tells the ratio between center to total distance of iris"""
    center_to_right_distance = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_distance / total_distance
    iris_pos = ""
    if ratio <= 0.22:
        iris_pos = "right"
    elif 0.22 < ratio <= 0.27:
        iris_pos = "center"
    else:
        iris_pos = "left"
    return iris_pos, ratio


def eye_scroller(scroll_landmarks):
    """Changes mouse position according to the eye position"""
    for id, land in enumerate(scroll_landmarks):
        if id == 1:
            screen_x = screen_w * land.x
            screen_y = screen_h * land.y
            pyautogui.moveTo(screen_x, screen_y)


def blink_click(eye, img_w, img_h):
    """Clicks if the left eye blinks."""
    for land in eye:
        x = int(img_w * land.x)
        y = int(img_h * land.y)
        cv.circle(frame, (x, y), 3, (0, 255, 255))
        eye_wink = (eye[0].y - eye[1].y) * 100
        if eye_wink < EYE_MESH_LANDMARKS.LEFT_EYE_CLOSE_THRESHOLD.value:
            print("click")
            pyautogui.click()
            pyautogui.sleep(1)


# Generates a face mesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # flips the camera frame
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            eye_scroller(results.multi_face_landmarks[0].landmark[474:478])
            left = [results.multi_face_landmarks[0].landmark[145],
                    results.multi_face_landmarks[0].landmark[159]]
            blink_click(left, img_w, img_h)
            mesh_points = np.array([np.multiply([p.x, p.y],
                                                [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[EYE_MESH_LANDMARKS.LEFT_IRIS.value])
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[EYE_MESH_LANDMARKS.RIGHT_IRIS.value])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame,
                      center_left,
                      int(l_rad),
                      (0, 255, 0),
                      1,
                      cv.LINE_AA)
            cv.circle(frame,
                      center_right,
                      int(r_rad),
                      (0, 255, 0),
                      1,
                      cv.LINE_AA)
            cv.circle(frame,
                      mesh_points[EYE_MESH_LANDMARKS.R_H_RIGHT.value[0]],
                      3,
                      (255, 255, 255),
                      -1,
                      cv.LINE_AA)
            cv.circle(frame,
                      mesh_points[EYE_MESH_LANDMARKS.R_H_LEFT.value[0]],
                      3,
                      (0, 255, 255),
                      -1,
                      cv.LINE_AA)
            iris_pos, ratio = iris_position(center_right,
                                            mesh_points[EYE_MESH_LANDMARKS.R_H_RIGHT.value],
                                            mesh_points[EYE_MESH_LANDMARKS.L_H_RIGHT.value[0]])
            cv.putText(frame,
                       f"Iris Position: {iris_pos}, "
                       f"{ratio: .2f}",
                       (30, 30),
                       cv.FONT_HERSHEY_PLAIN,
                       1.2,
                       (0, 255, 0),
                       1,
                       cv.LINE_AA)

        cv.imshow('Camera', frame)
        key = cv.waitKey(1)

        # terminates on pressing 'q'
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
