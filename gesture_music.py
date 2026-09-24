import cv2
import mediapipe as mp
import pygame
import time

"""
gesture music control

"""
# -----------------------------
# 1. Initialize pygame sound
# -----------------------------

pygame.mixer.init()

# Frequencies for five notes
notes = {
    1: 261,   # C4
    2: 294,   # D4
    3: 330,   # E4
    4: 349,   # F4
    5: 392    # G4
}


# -----------------------------
# 2. Create simple sounds
# -----------------------------

# def create_tone(frequency, duration=0.25):
#     sample_rate = 44100
#
#     import numpy as np
#
#     t = np.linspace(
#         0,
#         duration,
#         int(sample_rate * duration),
#         False
#     )
#
#     wave = np.sin(2 * np.pi * frequency * t)
#
#     audio = np.int16(wave * 32767)
#
#     stereo_audio = np.column_stack((audio, audio))
#
#     return pygame.sndarray.make_sound(stereo_audio)

def create_tone(frequency, duration=0.8):
    import numpy as np

    sample_rate = 44100

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        False
    )

    # Piano-like sound:
    # fundamental + several harmonics
    wave = (
        1.00 * np.sin(2 * np.pi * frequency * t)
        + 0.50 * np.sin(2 * np.pi * frequency * 2 * t)
        + 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
        + 0.12 * np.sin(2 * np.pi * frequency * 4 * t)
    )

    # Piano-like decay:
    # loud at the beginning, gradually becoming quieter
    envelope = np.exp(-3.5 * t)

    wave = wave * envelope

    # Normalize
    wave = wave / np.max(np.abs(wave))

    audio = np.int16(wave * 32767)

    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)

sounds = {}

for number, frequency in notes.items():
    sounds[number] = create_tone(frequency)


# -----------------------------
# 3. MediaPipe Hand Landmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Change this to your model file
model_path = "hand_landmarker.task"


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# -----------------------------
# 4. Count fingers
# -----------------------------

def count_fingers(landmarks):

    fingers = 0

    # Index finger
    if landmarks[8].y < landmarks[6].y:
        fingers += 1

    # Middle finger
    if landmarks[12].y < landmarks[10].y:
        fingers += 1

    # Ring finger
    if landmarks[16].y < landmarks[14].y:
        fingers += 1

    # Pinky
    if landmarks[20].y < landmarks[18].y:
        fingers += 1

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers += 1

    return fingers


# -----------------------------
# 5. Start webcam
# -----------------------------

cap = cv2.VideoCapture(0)

last_gesture = 0
last_time = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Flip image so it behaves like a mirror
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
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

        gesture = 0

        # -----------------------------
        # 6. Process detected hand
        # -----------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            gesture = count_fingers(hand)

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # -----------------------------
        # 7. Play music
        # -----------------------------

        current_time = time.time()

        if gesture != last_gesture:

            if gesture in sounds:

                sounds[gesture].play()

                print(
                    f"Gesture: {gesture} fingers → "
                    f"Note {gesture}"
                )

            elif gesture == 0:

                print("Fist / no fingers")

            last_gesture = gesture
            last_time = current_time

        # -----------------------------
        # 8. Display gesture
        # -----------------------------

        if gesture == 0:
            text = "Fist / No fingers"

        else:
            text = f"{gesture} fingers"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "1=C  2=D  3=E  4=F  5=G",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Gesture Music Controller",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
pygame.quit()