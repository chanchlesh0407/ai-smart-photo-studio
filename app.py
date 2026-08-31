import cv2
import mediapipe as mp
import time
import os
import subprocess
import sys

from gesture import recognize_gesture

from filter_engine import (
    apply_filter,
    get_filters,
    get_filter_count,
    get_filter_display_name
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

HAND_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

CAPTURED_FOLDER = os.path.join(
    BASE_DIR,
    "captured"
)

os.makedirs(
    CAPTURED_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK MODELS
# ============================================================

if not os.path.exists(FACE_MODEL):

    print("ERROR: Face model not found:")
    print(FACE_MODEL)
    sys.exit()


if not os.path.exists(HAND_MODEL):

    print("ERROR: Hand model not found:")
    print(HAND_MODEL)
    sys.exit()


# ============================================================
# IMAGE EDITOR
# ============================================================

def open_image_editor():

    editor_path = os.path.join(
        BASE_DIR,
        "image_editor.py"
    )

    if not os.path.exists(editor_path):

        print(
            "ERROR: image_editor.py not found."
        )

        return

    try:

        subprocess.Popen(
            [
                sys.executable,
                editor_path
            ],
            cwd=BASE_DIR
        )

        print(
            "Image Editor opened."
        )

    except Exception as e:

        print(
            "Could not open Image Editor:",
            e
        )


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


# ============================================================
# FACE OPTIONS
# ============================================================

face_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=FACE_MODEL
    ),

    running_mode=(
        VisionRunningMode.VIDEO
    ),

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# HAND OPTIONS
# ============================================================

hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),

    running_mode=(
        VisionRunningMode.VIDEO
    ),

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Webcam could not be opened."
    )

    sys.exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = (
    "AI Smart Photo Studio"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)


# ============================================================
# FILTER STATE
# ============================================================

current_category = "GLASSES"

current_index = 0

current_filter = "NONE"


# ============================================================
# SELECT CATEGORY
# ============================================================

def select_category(category):

    global current_category
    global current_index
    global current_filter

    filters = get_filters(
        category
    )

    current_category = category

    current_index = 0

    if filters:

        current_filter = filters[0]

        print(
            f"[GESTURE] Category: {category}"
        )

        print(
            f"[FILTER] {get_filter_display_name(current_filter)}"
        )

    else:

        current_filter = "NONE"

        print(
            f"[FILTER] No filters in {category}"
        )


# ============================================================
# NEXT FILTER
# ============================================================

def next_filter():

    global current_index
    global current_filter

    filters = get_filters(
        current_category
    )

    if not filters:

        current_filter = "NONE"

        return


    current_index += 1


    if current_index >= len(filters):

        current_index = 0


    current_filter = filters[
        current_index
    ]


    print(
        f"[FILTER] "
        f"{get_filter_display_name(current_filter)} "
        f"({current_index + 1}/{len(filters)})"
    )


# ============================================================
# PREVIOUS FILTER
# ============================================================

def previous_filter():

    global current_index
    global current_filter

    filters = get_filters(
        current_category
    )

    if not filters:

        current_filter = "NONE"

        return


    current_index -= 1


    if current_index < 0:

        current_index = (
            len(filters) - 1
        )


    current_filter = filters[
        current_index
    ]


    print(
        f"[FILTER] "
        f"{get_filter_display_name(current_filter)} "
        f"({current_index + 1}/{len(filters)})"
    )


# ============================================================
# GESTURE ACTION
# ============================================================

def handle_gesture(
    gesture
):

    global current_filter

    # --------------------------------------------------------
    # ONE FINGER
    # --------------------------------------------------------
    # Select glasses category
    # and cycle to next glasses

    if gesture == "ONE":

        if current_category == "GLASSES":

            next_filter()

        else:

            select_category(
                "GLASSES"
            )


    # --------------------------------------------------------
    # TWO FINGERS
    # --------------------------------------------------------
    # Select head category
    # and cycle head filters

    elif gesture == "TWO":

        if current_category == "HEAD":

            next_filter()

        else:

            select_category(
                "HEAD"
            )


    # --------------------------------------------------------
    # THREE FINGERS
    # --------------------------------------------------------

    elif gesture == "THREE":

        if current_category == "FACE":

            next_filter()

        else:

            select_category(
                "FACE"
            )


    # --------------------------------------------------------
    # THUMBS UP
    # --------------------------------------------------------

    elif gesture == "THUMBS UP":

        if current_category == "EFFECTS":

            next_filter()

        else:

            select_category(
                "EFFECTS"
            )


    # --------------------------------------------------------
    # OPEN PALM
    # --------------------------------------------------------

    elif gesture == "OPEN PALM":

        current_filter = "NONE"

        print(
            "[GESTURE] Filters removed"
        )


    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------
    # Capture is handled separately

    elif gesture == "FIST":

        capture_photo()


# ============================================================
# CAPTURE
# ============================================================

capture_count = 0

last_capture_time = 0

CAPTURE_COOLDOWN = 1.5

clean_frame = None


def capture_photo():

    global capture_count
    global last_capture_time

    if clean_frame is None:

        return


    current_time = time.time()


    if (
        current_time
        -
        last_capture_time
        <
        CAPTURE_COOLDOWN
    ):

        return


    capture_count += 1


    filename = (
        f"photo_{capture_count}.jpg"
    )


    save_path = os.path.join(
        CAPTURED_FOLDER,
        filename
    )


    cv2.imwrite(
        save_path,
        clean_frame
    )


    last_capture_time = (
        current_time
    )


    print()
    print(
        "================================"
    )
    print(
        "PHOTO CAPTURED"
    )
    print(
        save_path
    )
    print(
        "================================"
    )
    print()


# ============================================================
# GESTURE STABILITY
# ============================================================

last_gesture = "NONE"

gesture_start_time = 0

gesture_triggered = False

GESTURE_STABLE_TIME = 0.45


# ============================================================
# START
# ============================================================

print()
print(
    "================================================"
)
print(
    "           AI SMART PHOTO STUDIO"
)
print(
    "================================================"
)
print()
print(
    "GESTURE CONTROLS"
)
print()
print(
    "ONE          -> Glasses"
)
print(
    "TWO          -> Head"
)
print(
    "THREE        -> Face"
)
print(
    "THUMBS UP    -> Effects"
)
print(
    "OPEN PALM    -> Remove"
)
print(
    "FIST         -> Capture"
)
print()
print(
    "KEYBOARD"
)
print(
    "1 -> Glasses"
)
print(
    "2 -> Head"
)
print(
    "3 -> Face"
)
print(
    "4 -> Effects"
)
print(
    "N -> Next"
)
print(
    "P -> Previous"
)
print(
    "0 -> Remove"
)
print(
    "C -> Capture"
)
print(
    "E -> Editor"
)
print(
    "Q -> Quit"
)
print()
print(
    "================================================"
)
print()


# ============================================================
# MEDIAPIPE
# ============================================================

with FaceLandmarker.create_from_options(
    face_options
) as face_landmarker, \
     HandLandmarker.create_from_options(
         hand_options
     ) as hand_landmarker:


    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = cap.read()

        if not success:

            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        height, width = (
            frame.shape[:2]
        )


        # ====================================================
        # RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp = int(
            time.time() * 1000
        )


        # ====================================================
        # FACE
        # ====================================================

        face_result = (
            face_landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )


        # ====================================================
        # HAND
        # ====================================================

        hand_result = (
            hand_landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )


        # ====================================================
        # GESTURE DETECTION
        # ====================================================

        gesture = "NO HAND"


        if hand_result.hand_landmarks:

            hand = (
                hand_result.hand_landmarks[0]
            )


            try:

                detected = (
                    recognize_gesture(
                        hand
                    )
                )


                if detected:

                    gesture = detected


            except Exception as e:

                gesture = "UNKNOWN"


        # ====================================================
        # STABLE GESTURE LOGIC
        # ====================================================

        if gesture != last_gesture:

            last_gesture = gesture

            gesture_start_time = (
                time.time()
            )

            gesture_triggered = False


        else:

            elapsed = (
                time.time()
                -
                gesture_start_time
            )


            if (
                elapsed
                >=
                GESTURE_STABLE_TIME
                and
                not gesture_triggered
            ):

                if gesture not in (
                    "NO HAND",
                    "UNKNOWN"
                ):

                    handle_gesture(
                        gesture
                    )


                gesture_triggered = True


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # ====================================================
        # KEYBOARD CATEGORY
        # ====================================================

        if key == ord("1"):

            select_category(
                "GLASSES"
            )


        elif key == ord("2"):

            select_category(
                "HEAD"
            )


        elif key == ord("3"):

            select_category(
                "FACE"
            )


        elif key == ord("4"):

            select_category(
                "EFFECTS"
            )


        # ====================================================
        # NEXT / PREVIOUS
        # ====================================================

        elif key == ord("n"):

            next_filter()


        elif key == ord("p"):

            previous_filter()


        # ====================================================
        # REMOVE
        # ====================================================

        elif key == ord("0"):

            current_filter = "NONE"


        # ====================================================
        # CAPTURE
        # ====================================================

        elif key == ord("c"):

            capture_photo()


        # ====================================================
        # EDITOR
        # ====================================================

        elif key == ord("e"):

            open_image_editor()


        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            break


        # ====================================================
        # CLEAN FRAME
        # ====================================================

        clean_frame = frame.copy()


        # ====================================================
        # APPLY FILTER
        # ====================================================

        if face_result.face_landmarks:

            for face in (
                face_result.face_landmarks
            ):

                clean_frame = apply_filter(
                    clean_frame,
                    face,
                    current_filter
                )


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        display_frame = (
            clean_frame.copy()
        )


        # ====================================================
        # FACE TRACKING
        # ====================================================

        if face_result.face_landmarks:

            for face in (
                face_result.face_landmarks
            ):

                for landmark in face:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    if (
                        0 <= x < width
                        and
                        0 <= y < height
                    ):

                        cv2.circle(
                            display_frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )


        # ====================================================
        # HAND TRACKING
        # ====================================================

        if hand_result.hand_landmarks:

            hand = (
                hand_result.hand_landmarks[0]
            )


            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                if (
                    0 <= x < width
                    and
                    0 <= y < height
                ):

                    cv2.circle(
                        display_frame,
                        (x, y),
                        4,
                        (255, 0, 0),
                        -1
                    )


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


            for start, end in connections:

                x1 = int(
                    hand[start].x * width
                )

                y1 = int(
                    hand[start].y * height
                )

                x2 = int(
                    hand[end].x * width
                )

                y2 = int(
                    hand[end].y * height
                )


                cv2.line(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        panel_x1 = 15
        panel_y1 = 15

        panel_x2 = 350
        panel_y2 = 170


        cv2.rectangle(
            display_frame,
            (
                panel_x1,
                panel_y1
            ),
            (
                panel_x2,
                panel_y2
            ),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(
            display_frame,
            "AI SMART PHOTO STUDIO",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # CATEGORY
        # ====================================================

        cv2.putText(
            display_frame,
            f"Category: {current_category}",
            (25, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # FILTER NAME
        # ====================================================

        filter_name = (
            get_filter_display_name(
                current_filter
            )
        )


        count = get_filter_count(
            current_category
        )


        if count > 0:

            filter_text = (
                f"Filter: {filter_name} "
                f"({current_index + 1}/{count})"
            )

        else:

            filter_text = (
                "Filter: NONE"
            )


        cv2.putText(
            display_frame,
            filter_text,
            (25, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # GESTURE
        # ====================================================

        cv2.putText(
            display_frame,
            f"Gesture: {gesture}",
            (25, 119),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            display_frame,
            "1-4 Category | N/P Change",
            (25, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            display_frame,
            "C Capture | E Editor | Q Quit",
            (25, 161),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print(
    "AI Smart Photo Studio closed."
)

print(
    f"Photos captured: {capture_count}"
)