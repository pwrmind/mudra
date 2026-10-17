import cv2
import mediapipe as mp
# Явно импортируем внутренние модули, чтобы обойти ошибку AttributeError
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

import pyautogui
import math
import ctypes

# 1. Исправление масштабирования Windows
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

pyautogui.PAUSE = 0

# 2. Инициализация MediaPipe Hands напрямую из импортированного модуля
hands = mp_hands.Hands(
    max_num_hands=1,            
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7   
)

# 3. Настройки экрана и сглаживания
screen_w, screen_h = pyautogui.size()
SMOOTHING = 4  
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0
is_clicked = False

# 4. Запуск захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

print("Mudra MVP запущен.")
print("Для выхода нажмите ESC в окне видео.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Точка №8 (Указательный) и Точка №4 (Большой)
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]

            # Переводим в пиксели экрана
            target_x = int(index_finger.x * screen_w)
            target_y = int(index_finger.y * screen_h)

            # Сглаживание движения
            cloc_x = ploc_x + (target_x - ploc_x) / SMOOTHING
            cloc_y = ploc_y + (target_y - ploc_y) / SMOOTHING

            pyautogui.moveTo(int(cloc_x), int(cloc_y))
            ploc_x, ploc_y = cloc_x, cloc_y

            # Расстояние между пальцами
            distance = math.hypot((thumb_finger.x - index_finger.x) * w, 
                                  (thumb_finger.y - index_finger.y) * h)

            # Жест клика (мудра "Щепоть")
            if distance < 25:
                if not is_clicked:
                    pyautogui.click()
                    is_clicked = True
                    cv2.circle(frame, (int(index_finger.x * w), int(index_finger.y * h)), 15, (0, 255, 0), cv2.FILLED)
            else:
                is_clicked = False

            # Визуализация
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Mudra MVP', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
