import cv2
import mediapipe as mp
import numpy as np

"""
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
"""

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Create Hand Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

# Store previous fingertip position
previous_point = None

# Drawing canvas
canvas = None

frame_timestamp = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip webcam so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Create drawing canvas once
    if canvas is None:
        canvas = np.zeros_like(frame)

    # OpenCV uses BGR
    # MediaPipe expects an RGB image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    frame_timestamp += 33

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    # If a hand was detected
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Landmark #8 = index fingertip
        fingertip = hand[8]

        # Convert normalized coordinates to pixels
        x = int(fingertip.x * frame.shape[1])
        y = int(fingertip.y * frame.shape[0])

        current_point = (x, y)

        # Draw line from previous position to current position
        if previous_point is not None:
            cv2.line(
                canvas,
                previous_point,
                current_point,
                (0, 255, 0),
                5
            )

        previous_point = current_point

        # Show fingertip
        cv2.circle(
            frame,
            current_point,
            8,
            (0, 0, 255),
            -1
        )

    else:
        # No hand detected
        previous_point = None

    # Put drawing over webcam image
    output = cv2.add(frame, canvas)

    cv2.imshow("Air Drawing", output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()