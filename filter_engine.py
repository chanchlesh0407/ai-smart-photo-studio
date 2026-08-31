import cv2
import numpy as np
import os
import math


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILTER_DIR = os.path.join(
    BASE_DIR,
    "ar_filters"
)


# ============================================================
# FILTER FOLDERS
# ============================================================

FILTER_FOLDERS = {
    "GLASSES": os.path.join(FILTER_DIR, "glasses"),
    "HEAD": os.path.join(FILTER_DIR, "head"),
    "FACE": os.path.join(FILTER_DIR, "face"),
    "EFFECTS": os.path.join(FILTER_DIR, "effects")
}


# ============================================================
# FILTER STORAGE
# ============================================================

FILTERS = {
    "GLASSES": [],
    "HEAD": [],
    "FACE": [],
    "EFFECTS": []
}


IMAGE_CACHE = {}


# ============================================================
# DISCOVER FILTERS
# ============================================================

def discover_filters():

    for category in FILTERS:

        FILTERS[category].clear()

        folder = FILTER_FOLDERS[category]

        if not os.path.exists(folder):

            print(
                f"[FILTER] Folder not found: {folder}"
            )

            continue

        for filename in sorted(os.listdir(folder)):

            path = os.path.join(
                folder,
                filename
            )

            if not os.path.isfile(path):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp"
            ):
                continue

            filter_id = (
                category
                + "_"
                + os.path.splitext(filename)[0].upper()
            )

            FILTERS[category].append(
                filter_id
            )

    print()
    print("==============================================")
    print("             AR FILTER LIBRARY")
    print("==============================================")

    for category in FILTERS:

        print(
            f"{category}: "
            f"{len(FILTERS[category])} filter(s)"
        )

        for filter_id in FILTERS[category]:

            print(
                "   ",
                filter_id
            )

    print("==============================================")
    print()


# Load all available filters
discover_filters()


# ============================================================
# GET FILTER PATH
# ============================================================

def get_filter_path(filter_id):

    for category in FILTERS:

        if filter_id in FILTERS[category]:

            filename = filter_id[
                len(category) + 1:
            ]

            folder = FILTER_FOLDERS[
                category
            ]

            for file in os.listdir(folder):

                name = os.path.splitext(
                    file
                )[0].upper()

                if name == filename:

                    return os.path.join(
                        folder,
                        file
                    )

    return None


# ============================================================
# LOAD FILTER IMAGE
# ============================================================

def load_filter(filter_id):

    if filter_id in IMAGE_CACHE:

        return IMAGE_CACHE[
            filter_id
        ]

    path = get_filter_path(
        filter_id
    )

    if path is None:

        print(
            f"[FILTER ERROR] Cannot find: "
            f"{filter_id}"
        )

        return None

    image = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        print(
            f"[FILTER ERROR] Cannot load: "
            f"{path}"
        )

        return None


    # --------------------------------------------------------
    # Convert image to BGRA
    # --------------------------------------------------------

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGRA
        )

    elif image.shape[2] == 3:

        alpha = np.full(
            (
                image.shape[0],
                image.shape[1]
            ),
            255,
            dtype=np.uint8
        )

        image = np.dstack(
            (
                image,
                alpha
            )
        )


    IMAGE_CACHE[
        filter_id
    ] = image


    return image


# ============================================================
# MEDIAPIPE LANDMARKS
# ============================================================

LEFT_EYE = 33

RIGHT_EYE = 263

FOREHEAD = 10

NOSE = 1

CHIN = 152

LEFT_CHEEK = 234

RIGHT_CHEEK = 454


# ============================================================
# GET LANDMARK POINT
# ============================================================

def get_point(
    face,
    index,
    width,
    height
):

    landmark = face[index]

    return (
        int(
            landmark.x * width
        ),
        int(
            landmark.y * height
        )
    )


# ============================================================
# DISTANCE
# ============================================================

def distance(
    p1,
    p2
):

    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


# ============================================================
# FACE ROTATION ANGLE
# ============================================================

def get_angle(
    left_eye,
    right_eye
):

    dx = (
        right_eye[0]
        -
        left_eye[0]
    )

    dy = (
        right_eye[1]
        -
        left_eye[1]
    )

    return math.degrees(
        math.atan2(
            dy,
            dx
        )
    )


# ============================================================
# RESIZE FILTER
# ============================================================

def resize_filter(
    image,
    target_width
):

    h, w = image.shape[:2]

    if w <= 0:

        return image

    scale = (
        target_width
        /
        float(w)
    )

    target_height = max(
        1,
        int(
            h * scale
        )
    )

    return cv2.resize(
        image,
        (
            int(target_width),
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# ROTATE FILTER
# ============================================================

def rotate_filter(
    image,
    angle
):

    h, w = image.shape[:2]

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(
        matrix[0, 0]
    )

    sin = abs(
        matrix[0, 1]
    )

    new_w = int(
        h * sin
        +
        w * cos
    )

    new_h = int(
        h * cos
        +
        w * sin
    )

    matrix[0, 2] += (
        new_w / 2
        -
        center[0]
    )

    matrix[1, 2] += (
        new_h / 2
        -
        center[1]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            new_w,
            new_h
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
            0
        )
    )


# ============================================================
# ALPHA OVERLAY
# ============================================================

def overlay(
    background,
    foreground,
    x,
    y
):

    if foreground is None:

        return background

    bg_h, bg_w = (
        background.shape[:2]
    )

    fg_h, fg_w = (
        foreground.shape[:2]
    )


    # --------------------------------------------------------
    # Clip coordinates
    # --------------------------------------------------------

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        bg_w,
        x + fg_w
    )

    y2 = min(
        bg_h,
        y + fg_h
    )


    if (
        x1 >= x2
        or
        y1 >= y2
    ):

        return background


    fx1 = x1 - x

    fy1 = y1 - y

    fx2 = fx1 + (
        x2 - x1
    )

    fy2 = fy1 + (
        y2 - y1
    )


    fg = foreground[
        fy1:fy2,
        fx1:fx2
    ]


    if fg.shape[2] != 4:

        return background


    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------

    alpha = (
        fg[:, :, 3]
        .astype(
            np.float32
        )
        /
        255.0
    )


    alpha = alpha[:, :, None]


    fg_rgb = (
        fg[:, :, :3]
        .astype(
            np.float32
        )
    )


    bg_rgb = (
        background[
            y1:y2,
            x1:x2
        ]
        .astype(
            np.float32
        )
    )


    result = (
        fg_rgb * alpha
        +
        bg_rgb * (
            1.0 - alpha
        )
    )


    background[
        y1:y2,
        x1:x2
    ] = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


    return background


# ============================================================
# GET CATEGORY
# ============================================================

def get_category(
    filter_id
):

    if filter_id is None:

        return None

    for category in FILTERS:

        if filter_id in FILTERS[category]:

            return category

    return None


# ============================================================
# FILTER DISPLAY NAME
# ============================================================

def get_filter_display_name(
    filter_id
):

    if not filter_id:

        return "NONE"

    if "_" in filter_id:

        name = filter_id.split(
            "_",
            1
        )[1]

    else:

        name = filter_id


    return name.replace(
        "_",
        " "
    ).title()


# ============================================================
# GET FILTERS
# ============================================================

def get_filters(
    category
):

    return FILTERS.get(
        category,
        []
    )


# ============================================================
# GET FILTER COUNT
# ============================================================

def get_filter_count(
    category
):

    return len(
        FILTERS.get(
            category,
            []
        )
    )


# ============================================================
# APPLY FILTER
# ============================================================

def apply_filter(
    frame,
    face,
    filter_id
):

    if (
        face is None
        or
        filter_id is None
        or
        filter_id == "NONE"
    ):

        return frame


    category = get_category(
        filter_id
    )


    if category is None:

        return frame


    image = load_filter(
        filter_id
    )


    if image is None:

        return frame


    height, width = (
        frame.shape[:2]
    )


    # ========================================================
    # LANDMARKS
    # ========================================================

    left_eye = get_point(
        face,
        LEFT_EYE,
        width,
        height
    )


    right_eye = get_point(
        face,
        RIGHT_EYE,
        width,
        height
    )


    forehead = get_point(
        face,
        FOREHEAD,
        width,
        height
    )


    nose = get_point(
        face,
        NOSE,
        width,
        height
    )


    chin = get_point(
        face,
        CHIN,
        width,
        height
    )


    left_cheek = get_point(
        face,
        LEFT_CHEEK,
        width,
        height
    )


    right_cheek = get_point(
        face,
        RIGHT_CHEEK,
        width,
        height
    )


    # ========================================================
    # MEASUREMENTS
    # ========================================================

    eye_distance = distance(
        left_eye,
        right_eye
    )


    face_width = distance(
        left_cheek,
        right_cheek
    )


    angle = get_angle(
        left_eye,
        right_eye
    )


    # ========================================================
    # GLASSES
    # ========================================================

    if category == "GLASSES":

        # ----------------------------------------------------
        # Center of both eyes
        # ----------------------------------------------------

        center_x = (
            left_eye[0]
            +
            right_eye[0]
        ) / 2


        center_y = (
            left_eye[1]
            +
            right_eye[1]
        ) / 2


        # ----------------------------------------------------
        # GLASSES SIZE
        #
        # 1.65 = small
        # 1.95 = balanced
        # 2.35 = large
        # ----------------------------------------------------

        target_width = int(
            eye_distance * 2.2
        )


        # ----------------------------------------------------
        # FACE WIDTH LIMIT
        #
        # Prevent glasses from becoming
        # wider than the face.
        # ----------------------------------------------------

        maximum_width = int(
            face_width * 1.2
        )


        target_width = max(
            50,
            min(
                target_width,
                maximum_width
            )
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = resize_filter(
            image,
            target_width
        )


        # ----------------------------------------------------
        # Rotate according to head
        # ----------------------------------------------------

        image = rotate_filter(
            image,
            -angle
        )


        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        x = int(
            center_x
            -
            image.shape[1] / 2
        )


        y = int(
            center_y
            -
            image.shape[0] / 2
        )


    # ========================================================
    # HEAD FILTER
    # ========================================================

    elif category == "HEAD":

        center_x = (
            left_cheek[0]
            +
            right_cheek[0]
        ) / 2


        center_y = forehead[1]


        target_width = int(
            face_width * 1.5
        )


        target_width = max(
            60,
            min(
                target_width,
                int(width * 1.2)
            )
        )


        image = resize_filter(
            image,
            target_width
        )


        image = rotate_filter(
            image,
            -angle
        )


        x = int(
            center_x
            -
            image.shape[1] / 2
        )


        y = int(
            forehead[1]
            -
            image.shape[0]
            * 0.82
        )


    # ========================================================
    # FACE FILTER
    # ========================================================

    elif category == "FACE":

        center_x = (
            left_cheek[0]
            +
            right_cheek[0]
        ) / 2


        center_y = (
            forehead[1]
            +
            chin[1]
        ) / 2


        target_width = int(
            face_width * 1.1
        )


        target_width = max(
            60,
            min(
                target_width,
                int(width * .9)
            )
        )


        image = resize_filter(
            image,
            target_width
        )


        image = rotate_filter(
            image,
            -angle
        )


        x = int(
            center_x
            -
            image.shape[1] / 2
        )


        y = int(
            center_y
            -
            image.shape[0] / 2
        )


    # ========================================================
    # EFFECTS
    # ========================================================

    elif category == "EFFECTS":

        center_x = (
            left_eye[0]
            +
            right_eye[0]
        ) / 2


        center_y = (
            forehead[1]
            
        ) / 2


        target_width = int(
            face_width * 5
        )


        target_width = max(
            70,
            min(
                target_width,
                int(width * 0.85)
            )
        )


        image = resize_filter(
            image,
            target_width
        )


        x = int(
            center_x
            -
            image.shape[1] / 2
        )


        y = int(
            center_y
            -
            image.shape[0] / 2
        )


    else:

        return frame


    # ========================================================
    # APPLY
    # ========================================================

    return overlay(
        frame,
        image,
        x,
        y
    )


# ============================================================
# CLEAR CACHE
# ============================================================

def reset_filter_cache():

    IMAGE_CACHE.clear()