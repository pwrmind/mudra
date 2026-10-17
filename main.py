import os
# Отключаем лишние предупреждения TensorFlow Lite / Protobuf в консоли
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pyautogui
import math
import ctypes
import time
import warnings

# Глушим UserWarning от устаревшего Protobuf внутри MediaPipe
warnings.filterwarnings("ignore", category=UserWarning)

# Исправление масштабирования Windows
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

pyautogui.PAUSE = 0

# --- ONE EURO FILTER ---
class OneEuroFilter:
    def __init__(self, t0, x0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self.alpha(self.dcutoff, dt) * (dx - self.dx_prev)
        cutoff = self.mincutoff + self.beta * abs(edx)
        rx = self.x_prev + self.alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev = rx
        self.dx_prev = edx
        self.t_prev = t
        return rx

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

# Состояния для логики кликов и drag-and-drop
pinch_start_time = None
is_dragging = False

start_time = time.time()
filter_x = OneEuroFilter(start_time, screen_w / 2, mincutoff=0.05, beta=0.8)
filter_y = OneEuroFilter(start_time, screen_h / 2, mincutoff=0.05, beta=0.8)

cap = cv2.VideoCapture(0)
print("\nMudra с поддержкой Drag-and-Drop запущен.")
print("Для выхода нажмите ESC в окне видео.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # ИСПРАВЛЕНО: Явно берем нужные точки из массива по индексам
            index_finger = landmarks[8]  # Кончик указательного
            thumb_finger = landmarks[4]  # Кончик большого

            # Фильтрация координат курсора
            raw_x = index_finger.x * screen_w
            raw_y = index_finger.y * screen_h
            current_time = time.time()
            smooth_x = filter_x(current_time, raw_x)
            smooth_y = filter_y(current_time, raw_y)

            # Перемещаем мышь
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # Расстояние между пальцами для определения мудры
            distance = math.hypot((thumb_finger.x - index_finger.x) * w, 
                                  (thumb_finger.y - index_finger.y) * h)

            # ПАЛЬЦЫ СОЕДИНЕНЫ (Щепоть)
            if distance < 25:
                if pinch_start_time is None:
                    pinch_start_time = time.time()  # Фиксируем время начала щипка
                
                duration = time.time() - pinch_start_time

                # Если удерживаем дольше 0.3 секунд — включаем Drag
                if duration > 0.3:
                    if not is_dragging:
                        pyautogui.mouseDown()  # Зажимаем левую кнопку мыши
                        is_dragging = True
                    # Рисуем синий круг — режим перетаскивания активен
                    cv2.circle(frame, (int(index_finger.x * w), int(index_finger.y * h)), 20, (255, 0, 0), cv2.FILLED)
                else:
                    # Пока идет удержание до 0.3 сек — рисуем желтый круг (подготовка)
                    cv2.circle(frame, (int(index_finger.x * w), int(index_finger.y * h)), 15, (0, 255, 255), cv2.FILLED)

            # ПАЛЬЦЫ РАЗОМКНУТЫ
            else:
                if pinch_start_time is not None:
                    duration = time.time() - pinch_start_time
                    
                    if is_dragging:
                        # Если были в режиме перетаскивания — отпускаем объект
                        pyautogui.mouseUp()
                        is_dragging = False
                    elif duration < 0.3:
                        # Если щипок был быстрым — делаем обычный одиночный клик
                        pyautogui.click()
                        # Короткая визуализация клика (зеленый всплеск)
                        cv2.circle(frame, (int(index_finger.x * w), int(index_finger.y * h)), 25, (0, 255, 0), 2)
                    
                    # Сбрасываем таймер щипка
                    pinch_start_time = None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Безопасный сброс, если рука пропала из кадра во время перетаскивания
    else:
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
        pinch_start_time = None

    cv2.imshow('Mudra Drag & Drop', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
