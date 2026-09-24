import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
"""
gesture sign play
"""

# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360

NUM_PARTICLES = 650

MIN_SPEED = 2.6
MAX_SPEED = 4.5

SHAPE_DURATION = 3.0
SHAPE_TRANSITION_SPEED = 0.2

MODEL_PATH = "hand_landmarker.task"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# PARTICLE / LETTER CLASS
# ============================================================

class LetterParticle:

    def __init__(self):
        self.letter = random.choice(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        self.x = random.uniform(0, WINDOW_WIDTH)
        self.y = random.uniform(-WINDOW_HEIGHT, WINDOW_HEIGHT)

        self.speed = random.uniform(
            MIN_SPEED,
            MAX_SPEED
        )

        self.drift = random.uniform(-0.3, 0.3)

        self.size = random.randint(16, 30)

        self.angle = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-1, 1)

        self.target_x = self.x
        self.target_y = self.y

        self.original_x = self.x
        self.original_y = self.y

    # --------------------------------------------------------

    def snow_update(self):

        self.y += self.speed
        self.x += self.drift

        self.angle += self.rotation_speed

        # Wrap horizontally
        if self.x < -50:
            self.x = WINDOW_WIDTH + 50

        if self.x > WINDOW_WIDTH + 50:
            self.x = -50

        # Reappear at top
        if self.y > WINDOW_HEIGHT + 50:

            self.y = random.uniform(-100, -20)

            self.x = random.uniform(
                0,
                WINDOW_WIDTH
            )

            self.speed = random.uniform(
                MIN_SPEED,
                MAX_SPEED
            )

    # --------------------------------------------------------

    def move_to_target(self):

        self.x += (
            self.target_x - self.x
        ) * SHAPE_TRANSITION_SPEED

        self.y += (
            self.target_y - self.y
        ) * SHAPE_TRANSITION_SPEED

        self.angle += self.rotation_speed

    # --------------------------------------------------------

    def draw(self, frame):

        x = int(self.x)
        y = int(self.y)

        cv2.putText(
            frame,
            self.letter,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.size / 25.0,
            (245, 245, 255),
            1,
            cv2.LINE_AA
        )


# ============================================================
# CREATE PARTICLES
# ============================================================

particles = [
    LetterParticle()
    for _ in range(NUM_PARTICLES)
]


# ============================================================
# SHAPE GENERATORS
# ============================================================

def generate_octagon(cx, cy, radius, count):
    """
    Generate points around an octagon outline.
    """

    points = []

    vertices = []

    for i in range(8):

        angle = math.radians(
            -22.5 + i * 45
        )

        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        vertices.append((x, y))

    # Sample along edges
    for i in range(8):

        x1, y1 = vertices[i]

        x2, y2 = vertices[(i + 1) % 8]

        edge_count = max(
            1,
            count // 8
        )

        for j in range(edge_count):

            t = j / edge_count

            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            points.append((x, y))

    return points


# ------------------------------------------------------------

def generate_heart(cx, cy, scale, count):

    points = []

    t_values = np.linspace(
        0,
        2 * math.pi,
        count
    )

    for t in t_values:

        x = (
            16 * math.sin(t) ** 3
        )

        y = (
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )

        x = cx + x * scale
        y = cy - y * scale

        points.append((x, y))

    return points


# ------------------------------------------------------------

def generate_circle(cx, cy, radius, count):

    points = []

    for i in range(count):

        angle = (
            2 * math.pi * i / count
        )

        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        points.append((x, y))

    return points


# ------------------------------------------------------------

def generate_fish(cx, cy, scale, count):

    """
    Fish outline made from several curves.
    """

    points = []

    # Fish body
    body_points = []

    for i in range(count // 2):

        t = (
            2 * math.pi * i
            / (count // 2)
        )

        x = (
            math.cos(t) * 2.0
        )

        y = (
            math.sin(t) * 0.9
        )

        body_points.append(
            (
                cx + x * scale,
                cy + y * scale
            )
        )

    points.extend(body_points)

    # Tail
    tail_count = count - len(points)

    tail_points = [
        (
            cx - 2.0 * scale,
            cy
        ),

        (
            cx - 3.4 * scale,
            cy - 1.2 * scale
        ),

        (
            cx - 3.0 * scale,
            cy
        ),

        (
            cx - 3.4 * scale,
            cy + 1.2 * scale
        ),

        (
            cx - 2.0 * scale,
            cy
        )
    ]

    # Repeat/interpolate tail points
    for i in range(tail_count):

        a = i / max(1, tail_count - 1)

        p1 = tail_points[
            int(a * (len(tail_points) - 1))
        ]

        points.append(p1)

    return points


# ------------------------------------------------------------

def generate_crying_face(cx, cy, radius, count):

    """
    Crying face:
        circular outline
        eyes
        tears
        mouth
    """

    points = []

    # Face circle
    circle_count = int(count * 0.55)

    points.extend(
        generate_circle(
            cx,
            cy,
            radius,
            circle_count
        )
    )

    remaining = count - len(points)

    # Eyes
    eye_points = []

    left_eye_x = cx - radius * 0.35
    right_eye_x = cx + radius * 0.35

    eye_y = cy - radius * 0.2

    for ex in [left_eye_x, right_eye_x]:

        eye_points.append(
            (ex, eye_y)
        )

        eye_points.append(
            (ex, eye_y + radius * 0.1)
        )

    # Tears
    tear1 = [
        (
            left_eye_x,
            eye_y + radius * 0.12
        ),
        (
            left_eye_x - 5,
            eye_y + radius * 0.30
        ),
        (
            left_eye_x,
            eye_y + radius * 0.45
        ),
        (
            left_eye_x + 5,
            eye_y + radius * 0.30
        )
    ]

    tear2 = [
        (
            right_eye_x,
            eye_y + radius * 0.12
        ),
        (
            right_eye_x - 5,
            eye_y + radius * 0.30
        ),
        (
            right_eye_x,
            eye_y + radius * 0.45
        ),
        (
            right_eye_x + 5,
            eye_y + radius * 0.30
        )
    ]

    # Sad mouth
    mouth = []

    for i in range(remaining):

        t = (
            2 * math.pi * i
            / max(1, remaining)
        )

        x = (
            cx
            + radius * 0.35 * math.cos(t)
        )

        y = (
            cy
            + radius * 0.25
            + radius * 0.12 * math.sin(t)
        )

        mouth.append((x, y))

    points.extend(eye_points)
    points.extend(tear1)
    points.extend(tear2)
    points.extend(mouth)

    return points


# ============================================================
# CREATE TARGET SHAPE
# ============================================================

def create_shape(shape_name):

    center_x = WINDOW_WIDTH // 2
    center_y = WINDOW_HEIGHT // 2

    if shape_name == "STOP":

        return generate_octagon(
            center_x,
            center_y,
            190,
            NUM_PARTICLES
        )

    elif shape_name == "LOVE":

        return generate_heart(
            center_x,
            center_y,
            10,
            NUM_PARTICLES
        )

    elif shape_name == "SAD":

        return generate_crying_face(
            center_x,
            center_y,
            170,
            NUM_PARTICLES
        )

    elif shape_name == "EAT":

        return generate_fish(
            center_x,
            center_y,
            65,
            NUM_PARTICLES
        )

    return []


# ============================================================
# APPLY TARGETS
# ============================================================

def activate_shape(shape_name):

    targets = create_shape(
        shape_name
    )

    if len(targets) == 0:
        return

    # We have more particles than target points.
    # Repeat target positions so ALL letters participate.
    for i, particle in enumerate(particles):

        target = targets[
            i % len(targets)
        ]

        # Add a little random offset.
        # This prevents the shape from looking
        # like a perfectly mathematical computer drawing.
        jitter = 3

        particle.target_x = (
            target[0]
            + random.uniform(-jitter, jitter)
        )

        particle.target_y = (
            target[1]
            + random.uniform(-jitter, jitter)
        )


# ============================================================
# FINGER FUNCTIONS
# ============================================================

def finger_extended(hand, tip, pip):

    return hand[tip].y < hand[pip].y


def is_palm_open(hand):

    index = finger_extended(
        hand, 8, 6
    )

    middle = finger_extended(
        hand, 12, 10
    )

    ring = finger_extended(
        hand, 16, 14
    )

    pinky = finger_extended(
        hand, 20, 18
    )

    return (
        index
        and middle
        and ring
        and pinky
    )


# ------------------------------------------------------------

def is_pinch(hand):

    thumb = hand[4]

    index = hand[8]

    dx = thumb.x - index.x
    dy = thumb.y - index.y

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    return distance < 0.07


# ------------------------------------------------------------

def is_thumb_up(hand):

    thumb_tip = hand[4]
    thumb_ip = hand[3]
    thumb_mcp = hand[2]

    # Thumb pointing upward
    thumb_up = (
        thumb_tip.y
        < thumb_ip.y
        < thumb_mcp.y
    )

    # Other fingers mostly folded
    index_folded = (
        hand[8].y > hand[6].y
    )

    middle_folded = (
        hand[12].y > hand[10].y
    )

    ring_folded = (
        hand[16].y > hand[14].y
    )

    pinky_folded = (
        hand[20].y > hand[18].y
    )

    return (
        thumb_up
        and index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )


# ------------------------------------------------------------

def is_thumb_down(hand):

    thumb_tip = hand[4]
    thumb_ip = hand[3]
    thumb_mcp = hand[2]

    thumb_down = (
        thumb_tip.y
        > thumb_ip.y
        > thumb_mcp.y
    )

    index_folded = (
        hand[8].y > hand[6].y
    )

    middle_folded = (
        hand[12].y > hand[10].y
    )

    ring_folded = (
        hand[16].y > hand[14].y
    )

    pinky_folded = (
        hand[20].y > hand[18].y
    )

    return (
        thumb_down
        and index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )


# ============================================================
# GESTURE CLASSIFIER
# ============================================================

def detect_gesture(hand):

    # Check pinch first
    if is_pinch(hand):
        return "EAT"

    # Open palm
    if is_palm_open(hand):
        return "STOP"

    # Thumb gestures
    if is_thumb_up(hand):
        return "LOVE"

    if is_thumb_down(hand):
        return "SAD"

    return "NONE"


# ============================================================
# DRAW CAMERA HAND
# ============================================================

def draw_hand_landmarks(
        frame,
        hand_landmarks
):

    h, w, _ = frame.shape

    # Draw connections
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    for a, b in connections:

        x1 = int(
            hand_landmarks[a].x * w
        )

        y1 = int(
            hand_landmarks[a].y * h
        )

        x2 = int(
            hand_landmarks[b].x * w
        )

        y2 = int(
            hand_landmarks[b].y * h
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (100, 220, 255),
            2
        )

    for landmark in hand_landmarks:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1
        )


# ============================================================
# GAME STATE
# ============================================================

current_shape = None

shape_start_time = 0

last_triggered_gesture = "NONE"

stable_gesture = "NONE"

gesture_candidate = "NONE"

candidate_start_time = 0

GESTURE_STABILITY_TIME = 0.03


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WINDOW_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    WINDOW_HEIGHT
)


if not cap.isOpened():

    print(
        "ERROR: Cannot open webcam."
    )

    landmarker.close()

    exit()


# ============================================================
# MAIN GAME LOOP
# ============================================================

start_time = time.time()

frame_number = 0


while True:

    success, frame = cap.read()

    if not success:
        print(
            "ERROR: Cannot read webcam."
        )
        break

    frame_number += 1

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    # Resize
    frame = cv2.resize(
        frame,
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Create dark game background
    # --------------------------------------------------------

    game = np.zeros(
        (
            WINDOW_HEIGHT,
            WINDOW_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    # Very dark blue-gray background
    game[:] = (
        12,
        18,
        30
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        time.time() * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    detected_gesture = "NONE"

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        draw_hand_landmarks(frame, hand)

        detected_gesture = detect_gesture(hand)

    # --------------------------------------------------------
    # Gesture stabilization
    # --------------------------------------------------------

    current_time = time.time()

    if detected_gesture != gesture_candidate:

        gesture_candidate = detected_gesture

        candidate_start_time = current_time

    else:

        if (
            current_time
            - candidate_start_time
            > GESTURE_STABILITY_TIME
        ):

            stable_gesture = gesture_candidate

    # --------------------------------------------------------
    # Trigger new shape
    # --------------------------------------------------------

    if (
        stable_gesture != "NONE"
        and stable_gesture
        != last_triggered_gesture
    ):

        current_shape = stable_gesture

        shape_start_time = current_time

        activate_shape(
            current_shape
        )

        last_triggered_gesture = (
            stable_gesture
        )

    # --------------------------------------------------------
    # Shape mode
    # --------------------------------------------------------

    if current_shape is not None:

        elapsed = (
            current_time
            - shape_start_time
        )

        for particle in particles:

            particle.move_to_target()

        # After a few seconds, return to snowfall
        if elapsed > SHAPE_DURATION:

            current_shape = None

            last_triggered_gesture = (
                stable_gesture
            )

            # Throw particles back into snowfall
            for particle in particles:

                particle.y = random.uniform(
                    -WINDOW_HEIGHT,
                    WINDOW_HEIGHT
                )

                particle.x = random.uniform(
                    0,
                    WINDOW_WIDTH
                )

                particle.speed = random.uniform(
                    MIN_SPEED,
                    MAX_SPEED
                )

    # --------------------------------------------------------
    # Snowfall mode
    # --------------------------------------------------------

    else:

        for particle in particles:

            particle.snow_update()

    # --------------------------------------------------------
    # Draw letters
    # --------------------------------------------------------

    for particle in particles:

        particle.draw(
            game
        )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    cv2.putText(
        game,
        "GESTURE MAGIC",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (220, 230, 255),
        2,
        cv2.LINE_AA
    )

    # Gesture display

    if stable_gesture == "STOP":

        message = "STOP"

    elif stable_gesture == "LOVE":

        message = "LOVE"

    elif stable_gesture == "SAD":

        message = "SAD"

    elif stable_gesture == "EAT":

        message = "EAT"

    else:

        message = "SNOWFALL"

    cv2.putText(
        game,
        message,
        (
            WINDOW_WIDTH - 230,
            50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Small camera preview
    # --------------------------------------------------------

    preview_width = 240
    preview_height = 135

    preview = cv2.resize(
        frame,
        (
            preview_width,
            preview_height
        )
    )

    # Put camera preview in bottom-right
    x1 = (
        WINDOW_WIDTH
        - preview_width
        - 20
    )

    y1 = (
        WINDOW_HEIGHT
        - preview_height
        - 20
    )

    game[
        y1:y1 + preview_height,
        x1:x1 + preview_width
    ] = preview

    cv2.rectangle(
        game,
        (
            x1,
            y1
        ),
        (
            x1 + preview_width,
            y1 + preview_height
        ),
        (180, 200, 220),
        2
    )

    cv2.putText(
        game,
        "HAND",
        (
            x1 + 8,
            y1 + 22
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        game,
        "Open palm = STOP     "
        "Thumb up = LOVE     "
        "Thumb down = SAD     "
        "Pinch = EAT",
        (
            30,
            WINDOW_HEIGHT - 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 180, 200),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Gesture Magic",
        game
    )

    # ESC to quit
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()