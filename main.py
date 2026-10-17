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

# Прицельно глушим ТОЛЬКО шум protobuf из MediaPipe, не трогая прочие UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

# Исправление масштабирования Windows
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

pyautogui.PAUSE = 0
# КРИТИЧНО: иначе при ведении курсора к (0,0) — например, к кнопке "Пуск" —
# PyAutoGUI выбросит FailSafeException и скрипт упадёт
pyautogui.FAILSAFE = False

# Настройки поведения
DEBUG_OVERLAY        = True   # поставить False для "продакшена" (без скелета и FPS)
PINCH_RATIO_THRESH   = 0.35   # порог щипка, нормированный на размер ладони
DRAG_HOLD_TIME       = 0.15   # сколько держать щипок перед тем, как разрешить drag
DRAG_MOVE_THRESHOLD  = 20     # минимальный сдвиг в пикселях экрана, чтобы стал drag
HAND_LOST_FRAMES     = 15     # ~0.5 сек при 30 FPS — сколько ждать перед mouseUp


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
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self.alpha(self.dcutoff, dt) * (dx - self.dx_prev)
        cutoff = self.mincutoff + self.beta * abs(edx)
        rx = self.x_prev + self.alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev = rx
        self.dx_prev = edx
        self.t_prev = t
        return rx


# --- Инициализация MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

# Состояния
pinch_start_time   = None
pinch_anchor       = None   # (x, y) на экране — где начался щипок
pinch_anchor_frame = None   # (x, y) в кадре — для визуализации
is_dragging        = False
hand_lost_frames   = 0

# Фильтры
start_time = time.time()
filter_x = OneEuroFilter(start_time, screen_w / 2, mincutoff=0.05, beta=0.8)
filter_y = OneEuroFilter(start_time, screen_h / 2, mincutoff=0.05, beta=0.8)

cap = cv2.VideoCapture(0)
print("\nMudra с поддержкой Drag-and-Drop запущен.")
print("Короткий щипок — клик. Долгий щипок + сдвиг — перетаскивание.")
print("Для выхода нажмите ESC в окне видео.\n")

prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_lost_frames = 0

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks    = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]
            wrist        = landmarks[0]
            middle_mcp   = landmarks[9]

            # Сглаживание координат
            raw_x = index_finger.x * screen_w
            raw_y = index_finger.y * screen_h
            smooth_x = filter_x(current_time, raw_x)
            smooth_y = filter_y(current_time, raw_y)

            # Нормируем расстояние на размер ладони — инвариант к дистанции до камеры
            palm_size = math.hypot(
                (wrist.x - middle_mcp.x) * w,
                (wrist.y - middle_mcp.y) * h,
            )
            pinch_dist = math.hypot(
                (thumb_finger.x - index_finger.x) * w,
                (thumb_finger.y - index_finger.y) * h,
            )
            pinch_ratio = pinch_dist / max(palm_size, 1e-6)
            is_pinching = pinch_ratio < PINCH_RATIO_THRESH

            # --- СОСТОЯНИЕ: ЩИПОК ---
            if is_pinching:
                if pinch_start_time is None:
                    # Фиксируем якорь — именно здесь произойдёт клик/захват
                    pinch_start_time   = current_time
                    pinch_anchor       = (int(smooth_x), int(smooth_y))
                    pinch_anchor_frame = (int(index_finger.x * w), int(index_finger.y * h))
                    # На всякий случай закрываем "зависший" drag с прошлого раза
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                duration  = current_time - pinch_start_time
                move_dist = math.hypot(
                    smooth_x - pinch_anchor[0],
                    smooth_y - pinch_anchor[1],
                )

                # Переход в режим drag: подержали И сдвинули
                if (not is_dragging
                        and duration  > DRAG_HOLD_TIME
                        and move_dist > DRAG_MOVE_THRESHOLD):
                    pyautogui.mouseDown()
                    is_dragging = True

                if is_dragging:
                    # Ведём объект за пальцем
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    if DEBUG_OVERLAY:
                        cv2.circle(frame, pinch_anchor_frame, 20, (255, 0, 0), cv2.FILLED)
                else:
                    # Курсор «примёрз» к якорю — щипок ещё не превратился в drag
                    pyautogui.moveTo(*pinch_anchor)
                    if DEBUG_OVERLAY:
                        cv2.circle(frame, pinch_anchor_frame, 15, (0, 255, 255), cv2.FILLED)

            # --- СОСТОЯНИЕ: НЕ ЩИПОК ---
            else:
                # В обычном режиме курсор свободно следует за пальцем
                if not is_dragging:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))

                if pinch_start_time is not None:
                    if is_dragging:
                        # Отпускаем перетаскиваемый объект
                        pyautogui.mouseUp()
                        is_dragging = False
                    else:
                        # Короткий или долгий щипок без сдвига — это клик по якорю
                        pyautogui.click(*pinch_anchor)
                        if DEBUG_OVERLAY:
                            cv2.circle(frame, pinch_anchor_frame, 25, (0, 255, 0), 2)
                    pinch_start_time   = None
                    pinch_anchor       = None
                    pinch_anchor_frame = None

            if DEBUG_OVERLAY:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- РУКА ПРОПАЛА ---
    else:
        hand_lost_frames += 1
        if hand_lost_frames == 1:
            # Первый кадр без руки — щипок недействителен
            pinch_start_time   = None
            pinch_anchor       = None
            pinch_anchor_frame = None
        # Drag держим до последнего: даём шанс вернуть руку в кадр
        if hand_lost_frames >= HAND_LOST_FRAMES and is_dragging:
            pyautogui.mouseUp()
            is_dragging = False

    # --- FPS / оверлей ---
    if DEBUG_OVERLAY:
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_dragging:
            status = "DRAG"
        elif pinch_start_time is not None:
            status = "HOLD"
        else:
            status = "MOVE"
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    prev_time = current_time

    cv2.imshow('Mudra Drag & Drop', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()