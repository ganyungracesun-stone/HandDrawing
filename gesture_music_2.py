import cv2
import mediapipe as mp
import pygame
import time
import numpy as np
"""
thumb:
index finger:
middle finger:
yep gesture:
pinch gesture:
"""

# ============================================================
# 1. Initialize sound
# ============================================================

pygame.mixer.init()

# Musical notes
notes = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "G": 392.00,
    "A": 440.00
}


# ============================================================
# 2. Create piano-like sounds
# ============================================================

def create_tone(frequency, duration=0.8):

    sample_rate = 44100

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        False
    )

    # Fundamental + harmonics
    wave = (
        1.00 * np.sin(2 * np.pi * frequency * t)
        + 0.50 * np.sin(2 * np.pi * frequency * 2 * t)
        + 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
        + 0.12 * np.sin(2 * np.pi * frequency * 4 * t)
    )

    # Piano-like decay
    envelope = np.exp(-3.5 * t)

    wave = wave * envelope

    # Normalize
    wave = wave / np.max(np.abs(wave))

    audio = np.int16(wave * 32767)

    stereo_audio = np.column_stack(
        (audio, audio)
    )

    return pygame.sndarray.make_sound(stereo_audio)


sounds = {}

for note, frequency in notes.items():
    sounds[note] = create_tone(frequency)


# ============================================================
# 3. MediaPipe Hand Landmarker
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


model_path = "hand_landmarker.task"


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=model_path
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1
)


# ============================================================
# 4. Gesture recognition
# ============================================================

def distance(p1, p2):

    return np.sqrt(
        (p1.x - p2.x) ** 2
        + (p1.y - p2.y) ** 2
    )


def recognize_gesture(landmarks):

    # --------------------------------------------------------
    # Important landmark numbers
    #
    # Thumb:
    # 4 = thumb tip
    # 3 = thumb joint
    #
    # Index:
    # 8 = index tip
    # 6 = index joint
    #
    # Middle:
    # 12 = middle tip
    # 10 = middle joint
    # --------------------------------------------------------

    thumb_tip = landmarks[4]
    thumb_joint = landmarks[3]

    index_tip = landmarks[8]
    index_joint = landmarks[6]

    middle_tip = landmarks[12]
    middle_joint = landmarks[10]

    # --------------------------------------------------------
    # Detect pinch
    # Thumb tip close to index tip
    # --------------------------------------------------------

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    pinch = pinch_distance < 0.07

    # --------------------------------------------------------
    # Finger positions
    # --------------------------------------------------------

    index_up = (
        index_tip.y < index_joint.y
    )

    middle_up = (
        middle_tip.y < middle_joint.y
    )

    thumb_up = (
        thumb_tip.x < thumb_joint.x
    )

    # ========================================================
    # Gesture rules
    # ========================================================

    # 🤏 Thumb + index pinch → A
    if pinch:

        return "A"

    # ✌️ Index + middle → G
    elif index_up and middle_up:

        return "G"

    # ☝️ Index only → D
    elif index_up and not middle_up:

        return "D"

    # 🖕 Middle only → E
    elif middle_up and not index_up:

        return "E"

    # 👍 Thumb only → C
    elif thumb_up:

        return "C"

    # No recognized gesture
    else:

        return None


# ============================================================
# 5. Start webcam
# ============================================================

cap = cv2.VideoCapture(0)

last_gesture = None

# Prevent accidental repeated notes
last_play_time = 0

# Minimum time between notes
cooldown = 0.25


# ============================================================
# 6. Run MediaPipe
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        gesture = None

        # ====================================================
        # Process hand
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            gesture = recognize_gesture(hand)

            # ------------------------------------------------
            # Draw landmarks
            # ------------------------------------------------

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # ====================================================
        # Play note
        # ====================================================

        current_time = time.time()

        if gesture is not None:

            # Only play when gesture changes
            # OR enough time has passed
            if (
                gesture != last_gesture
                and
                current_time - last_play_time > cooldown
            ):

                sounds[gesture].play()

                print(
                    f"Gesture: {gesture}"
                )

                last_play_time = current_time

        last_gesture = gesture

        # ====================================================
        # Display gesture
        # ====================================================

        if gesture == "C":

            text = "THUMB -> C"

        elif gesture == "D":

            text = "INDEX -> D"

        elif gesture == "E":

            text = "MIDDLE -> E"

        elif gesture == "G":

            text = "INDEX + MIDDLE -> G"

        elif gesture == "A":

            text = "PINCH -> A"

        else:

            text = "No gesture"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Thumb=C  Index=D  Middle=E",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Index+Middle=G  Pinch=A",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # Show webcam
        # ====================================================

        cv2.imshow(
            "Gesture Music Controller",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# 7. Clean up
# ============================================================

cap.release()

cv2.destroyAllWindows()

pygame.quit()
