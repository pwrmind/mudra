import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pyautogui
import math
import ctypes
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


# ============================================================
# ТУМБЛЕРЫ И ПАРАМЕТРЫ
# ============================================================

USE_PALM_HYBRID     = True
USE_DEAD_ZONE       = True
PRECISION_ON_PINCH  = True
DEBUG_OVERLAY       = True

PALM_MINCUTOFF      = 0.025
PALM_BETA           = 0.5
OFFSET_MINCUTOFF    = 0.12
OFFSET_BETA         = 1.2
DIRECT_MINCUTOFF    = 0.05
DIRECT_BETA         = 0.8
PRECISION_MINCUTOFF = 0.010

DEAD_ZONE_PX        = 2.5

# --- Жесты (исправлено) ---
PINCH_RATIO_THRESH  = 0.35
DRAG_HOLD_TIME      = 0.30   # ← назад к 0.3 (было 0.15 — клик превращался в drag)
DRAG_MOVE_THRESHOLD = 60     # ← 60 px, а не 20 (было слишком мелко)
HAND_LOST_FRAMES    = 15
REACQUIRE_RESET     = 10


# ============================================================
# ONE EURO FILTER
# ============================================================

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

    def reset(self, t, x):
        self.x_prev  = x
        self.dx_prev = 0.0
        self.t_prev  = t

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self.alpha(self.dcutoff, dt) * (dx - self.dx_prev)
        cutoff = self.mincutoff + self.beta * abs(edx)
        rx = self.x_prev + self.alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev  = rx
        self.dx_prev = edx
        self.t_prev  = t
        return rx


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

start_time = time.time()
filter_palm_x = OneEuroFilter(start_time, screen_w / 2, PALM_MINCUTOFF, PALM_BETA)
filter_palm_y = OneEuroFilter(start_time, screen_h / 2, PALM_MINCUTOFF, PALM_BETA)
filter_off_x  = OneEuroFilter(start_time, 0.0, OFFSET_MINCUTOFF, OFFSET_BETA)
filter_off_y  = OneEuroFilter(start_time, 0.0, OFFSET_MINCUTOFF, OFFSET_BETA)
filter_dir_x  = OneEuroFilter(start_time, screen_w / 2, DIRECT_MINCUTOFF, DIRECT_BETA)
filter_dir_y  = OneEuroFilter(start_time, screen_h / 2, DIRECT_MINCUTOFF, DIRECT_BETA)


# ============================================================
# КАМЕРА
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          60)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print("\nMudra запущен.")
print(f"Камера: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.0f} FPS")
print(f"DRAG_HOLD={DRAG_HOLD_TIME}s, DRAG_MOVE={DRAG_MOVE_THRESHOLD}px")
print("ESC — выход.\n")


# ============================================================
# СОСТОЯНИЯ
# ============================================================

pinch_start_time   = None
pinch_anchor       = None
pinch_anchor_frame = None
is_dragging        = False
hand_lost_frames   = 0

last_move_x, last_move_y = screen_w / 2, screen_h / 2

fps_smoothed = 0.0
prev_time    = time.time()


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

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
        was_lost = hand_lost_frames >= REACQUIRE_RESET
        hand_lost_frames = 0

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks    = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]
            wrist        = landmarks[0]
            middle_mcp   = landmarks[9]

            palm_x_norm = (landmarks[0].x + landmarks[5].x + landmarks[9].x
                           + landmarks[13].x + landmarks[17].x) / 5.0
            palm_y_norm = (landmarks[0].y + landmarks[5].y + landmarks[9].y
                           + landmarks[13].y + landmarks[17].y) / 5.0

            palm_px_x   = palm_x_norm * screen_w
            palm_px_y   = palm_y_norm * screen_h
            offset_px_x = (index_finger.x - palm_x_norm) * screen_w
            offset_px_y = (index_finger.y - palm_y_norm) * screen_h
            direct_px_x = index_finger.x * screen_w
            direct_px_y = index_finger.y * screen_h

            if was_lost:
                filter_palm_x.reset(current_time, palm_px_x)
                filter_palm_y.reset(current_time, palm_px_y)
                filter_off_x.reset(current_time, offset_px_x)
                filter_off_y.reset(current_time, offset_px_y)
                filter_dir_x.reset(current_time, direct_px_x)
                filter_dir_y.reset(current_time, direct_px_y)
                last_move_x, last_move_y = direct_px_x, direct_px_y

            in_precision = (PRECISION_ON_PINCH
                            and pinch_start_time is not None
                            and not is_dragging)
            if in_precision:
                filter_off_x.mincutoff = PRECISION_MINCUTOFF
                filter_off_y.mincutoff = PRECISION_MINCUTOFF
                filter_dir_x.mincutoff = PRECISION_MINCUTOFF
                filter_dir_y.mincutoff = PRECISION_MINCUTOFF
            else:
                filter_off_x.mincutoff = OFFSET_MINCUTOFF
                filter_off_y.mincutoff = OFFSET_MINCUTOFF
                filter_dir_x.mincutoff = DIRECT_MINCUTOFF
                filter_dir_y.mincutoff = DIRECT_MINCUTOFF

            if USE_PALM_HYBRID:
                palm_s_x = filter_palm_x(current_time, palm_px_x)
                palm_s_y = filter_palm_y(current_time, palm_px_y)
                off_s_x  = filter_off_x(current_time, offset_px_x)
                off_s_y  = filter_off_y(current_time, offset_px_y)
                smooth_x = palm_s_x + off_s_x
                smooth_y = palm_s_y + off_s_y
            else:
                smooth_x = filter_dir_x(current_time, direct_px_x)
                smooth_y = filter_dir_y(current_time, direct_px_y)

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

            # =============================================
            # ЖЕСТ: ЩИПОК
            # =============================================
            if is_pinching:
                if pinch_start_time is None:
                    pinch_start_time   = current_time
                    pinch_anchor       = (int(smooth_x), int(smooth_y))
                    pinch_anchor_frame = (int(index_finger.x * w),
                                          int(index_finger.y * h))
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                duration  = current_time - pinch_start_time
                move_dist = math.hypot(smooth_x - pinch_anchor[0],
                                       smooth_y - pinch_anchor[1])

                if (not is_dragging
                        and duration  > DRAG_HOLD_TIME
                        and move_dist > DRAG_MOVE_THRESHOLD):
                    pyautogui.mouseDown()
                    is_dragging = True

                if is_dragging:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    last_move_x, last_move_y = smooth_x, smooth_y
                    if DEBUG_OVERLAY:
                        cv2.circle(frame, pinch_anchor_frame, 20,
                                   (255, 0, 0), cv2.FILLED)
                else:
                    pyautogui.moveTo(*pinch_anchor)
                    last_move_x, last_move_y = pinch_anchor
                    if DEBUG_OVERLAY:
                        cv2.circle(frame, pinch_anchor_frame, 15,
                                   (0, 255, 255), cv2.FILLED)

            # =============================================
            # ЖЕСТ: НЕ ЩИПОК
            # =============================================
            else:
                # ИСПРАВЛЕНО: если drag активен, но щипок уже сброшен
                # (например, рука пропадала и вернулась разжатой) —
                # принудительно отпускаем мышь, иначе она «залипнет».
                if is_dragging and pinch_start_time is None:
                    pyautogui.mouseUp()
                    is_dragging = False

                if not is_dragging:
                    should_move = True
                    if USE_DEAD_ZONE:
                        if math.hypot(smooth_x - last_move_x,
                                      smooth_y - last_move_y) < DEAD_ZONE_PX:
                            should_move = False
                    if should_move:
                        pyautogui.moveTo(int(smooth_x), int(smooth_y))
                        last_move_x, last_move_y = smooth_x, smooth_y

                if pinch_start_time is not None:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    else:
                        pyautogui.click(*pinch_anchor)
                        if DEBUG_OVERLAY:
                            cv2.circle(frame, pinch_anchor_frame, 25,
                                       (0, 255, 0), 2)
                    pinch_start_time   = None
                    pinch_anchor       = None
                    pinch_anchor_frame = None

            if DEBUG_OVERLAY:
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

    else:
        hand_lost_frames += 1
        if hand_lost_frames == 1:
            pinch_start_time   = None
            pinch_anchor       = None
            pinch_anchor_frame = None
        if hand_lost_frames >= HAND_LOST_FRAMES and is_dragging:
            pyautogui.mouseUp()
            is_dragging = False

    if DEBUG_OVERLAY:
        dt_frame = max(current_time - prev_time, 1e-6)
        fps_now  = 1.0 / dt_frame
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps_now if fps_smoothed else fps_now

        cv2.putText(frame, f"FPS: {fps_smoothed:5.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_dragging:
            status, color = "DRAG", (255, 0, 0)
        elif pinch_start_time is not None:
            status, color = "HOLD", (0, 255, 255)
        else:
            status, color = "MOVE", (200, 200, 200)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        mode_bits = []
        if USE_PALM_HYBRID:    mode_bits.append("PALM")
        if USE_DEAD_ZONE:      mode_bits.append("DZ")
        if PRECISION_ON_PINCH: mode_bits.append("PREC")
        cv2.putText(frame, " ".join(mode_bits), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    prev_time = current_time

    cv2.imshow('Mudra Drag & Drop', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()