import math


# ============================================================
# DISTANCE BETWEEN TWO LANDMARKS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# 3D DISTANCE
# ============================================================

def distance_3d(p1, p2):

    x1 = getattr(p1, "x", 0.0)
    y1 = getattr(p1, "y", 0.0)
    z1 = getattr(p1, "z", 0.0)

    x2 = getattr(p2, "x", 0.0)
    y2 = getattr(p2, "y", 0.0)
    z2 = getattr(p2, "z", 0.0)

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2 +
        (z1 - z2) ** 2
    )


# ============================================================
# FINGER EXTENSION
# ============================================================

def finger_extended(
    landmarks,
    tip,
    pip,
    mcp,
    wrist=0
):

    tip_to_wrist = distance_3d(
        landmarks[tip],
        landmarks[wrist]
    )

    pip_to_wrist = distance_3d(
        landmarks[pip],
        landmarks[wrist]
    )

    mcp_to_wrist = distance_3d(
        landmarks[mcp],
        landmarks[wrist]
    )

    # Tip should be significantly farther
    # from wrist than PIP/MCP.
    return (
        tip_to_wrist
        >
        pip_to_wrist * 1.08
        and
        tip_to_wrist
        >
        mcp_to_wrist * 1.25
    )


# ============================================================
# FINGER CLOSED
# ============================================================

def finger_closed(
    landmarks,
    tip,
    pip,
    mcp
):

    tip_to_wrist = distance_3d(
        landmarks[tip],
        landmarks[0]
    )

    pip_to_wrist = distance_3d(
        landmarks[pip],
        landmarks[0]
    )

    mcp_to_wrist = distance_3d(
        landmarks[mcp],
        landmarks[0]
    )

    return (
        tip_to_wrist
        <
        pip_to_wrist * 1.08
        or
        tip_to_wrist
        <
        mcp_to_wrist * 1.20
    )


# ============================================================
# INDEX
# ============================================================

def index_up(landmarks):

    return finger_extended(
        landmarks,
        8,
        6,
        5
    )


# ============================================================
# MIDDLE
# ============================================================

def middle_up(landmarks):

    return finger_extended(
        landmarks,
        12,
        10,
        9
    )


# ============================================================
# RING
# ============================================================

def ring_up(landmarks):

    return finger_extended(
        landmarks,
        16,
        14,
        13
    )


# ============================================================
# PINKY
# ============================================================

def pinky_up(landmarks):

    return finger_extended(
        landmarks,
        20,
        18,
        17
    )


# ============================================================
# THUMB EXTENDED
# ============================================================

def thumb_extended(landmarks):

    thumb_tip = landmarks[4]

    thumb_ip = landmarks[3]

    thumb_mcp = landmarks[2]

    index_mcp = landmarks[5]

    wrist = landmarks[0]


    thumb_length = distance_3d(
        thumb_tip,
        thumb_mcp
    )


    palm_size = distance_3d(
        wrist,
        index_mcp
    )


    # Thumb must have reasonable length
    long_enough = (
        thumb_length
        >
        palm_size * 0.45
    )


    # Thumb tip should be away from palm
    tip_from_palm = distance_3d(
        thumb_tip,
        wrist
    )

    mcp_from_palm = distance_3d(
        thumb_mcp,
        wrist
    )


    away_from_palm = (
        tip_from_palm
        >
        mcp_from_palm * 1.18
    )


    return (
        long_enough
        and
        away_from_palm
    )


# ============================================================
# THUMB CLOSED
# ============================================================

def thumb_closed(landmarks):

    thumb_tip = landmarks[4]

    index_mcp = landmarks[5]

    wrist = landmarks[0]


    palm_size = distance_3d(
        wrist,
        index_mcp
    )


    thumb_distance = distance_3d(
        thumb_tip,
        wrist
    )


    return (
        thumb_distance
        <
        palm_size * 1.55
    )


# ============================================================
# FINGER COUNT
# ============================================================

def count_fingers(landmarks):

    count = 0


    if index_up(landmarks):
        count += 1


    if middle_up(landmarks):
        count += 1


    if ring_up(landmarks):
        count += 1


    if pinky_up(landmarks):
        count += 1


    return count


# ============================================================
# THUMBS UP
# ============================================================

def is_thumbs_up(landmarks):

    index = index_up(
        landmarks
    )

    middle = middle_up(
        landmarks
    )

    ring = ring_up(
        landmarks
    )

    pinky = pinky_up(
        landmarks
    )

    thumb = thumb_extended(
        landmarks
    )


    # Four fingers must be closed
    fingers_closed = (
        not index
        and
        not middle
        and
        not ring
        and
        not pinky
    )


    return (
        thumb
        and
        fingers_closed
    )


# ============================================================
# FIST
# ============================================================

def is_fist(landmarks):

    index = index_up(
        landmarks
    )

    middle = middle_up(
        landmarks
    )

    ring = ring_up(
        landmarks
    )

    pinky = pinky_up(
        landmarks
    )


    # All four fingers closed
    fingers_closed = (
        not index
        and
        not middle
        and
        not ring
        and
        not pinky
    )


    # A fist can have the thumb crossing
    # over the fingers, so don't require
    # a particular thumb position.

    return fingers_closed


# ============================================================
# ONE FINGER
# ============================================================

def is_one_finger(landmarks):

    return (
        index_up(landmarks)
        and
        not middle_up(landmarks)
        and
        not ring_up(landmarks)
        and
        not pinky_up(landmarks)
        and
        not is_thumbs_up(landmarks)
    )


# ============================================================
# TWO FINGERS
# ============================================================

def is_two_fingers(landmarks):

    return (
        index_up(landmarks)
        and
        middle_up(landmarks)
        and
        not ring_up(landmarks)
        and
        not pinky_up(landmarks)
    )


# ============================================================
# THREE FINGERS
# ============================================================

def is_three_fingers(landmarks):

    return (
        index_up(landmarks)
        and
        middle_up(landmarks)
        and
        ring_up(landmarks)
        and
        not pinky_up(landmarks)
    )


# ============================================================
# FOUR FINGERS
# ============================================================

def is_four_fingers(landmarks):

    return (
        index_up(landmarks)
        and
        middle_up(landmarks)
        and
        ring_up(landmarks)
        and
        pinky_up(landmarks)
        and
        not is_thumbs_up(landmarks)
    )


# ============================================================
# OPEN PALM
# ============================================================

def is_open_palm(landmarks):

    return (
        index_up(landmarks)
        and
        middle_up(landmarks)
        and
        ring_up(landmarks)
        and
        pinky_up(landmarks)
        and
        thumb_extended(landmarks)
    )


# ============================================================
# OK GESTURE
# ============================================================

def is_ok_gesture(landmarks):

    thumb_tip = landmarks[4]

    index_tip = landmarks[8]


    middle = middle_up(
        landmarks
    )

    ring = ring_up(
        landmarks
    )

    pinky = pinky_up(
        landmarks
    )


    # Distance between thumb and index
    pinch_distance = distance_3d(
        thumb_tip,
        index_tip
    )


    palm_size = distance_3d(
        landmarks[0],
        landmarks[9]
    )


    normalized_distance = (
        pinch_distance
        /
        max(
            palm_size,
            0.0001
        )
    )


    # Slightly stricter than previous
    pinch = (
        normalized_distance
        <
        0.42
    )


    # Other fingers should be extended
    other_fingers = (
        middle
        and
        ring
        and
        pinky
    )


    return (
        pinch
        and
        other_fingers
    )


# ============================================================
# PEACE / V SIGN
# ============================================================

def is_peace(landmarks):

    return (
        index_up(landmarks)
        and
        middle_up(landmarks)
        and
        not ring_up(landmarks)
        and
        not pinky_up(landmarks)
    )


# ============================================================
# MAIN RECOGNIZER
# ============================================================

def recognize_gesture(landmarks):

    # --------------------------------------------------------
    # OK
    # --------------------------------------------------------

    if is_ok_gesture(
        landmarks
    ):

        return "OK"


    # --------------------------------------------------------
    # THUMBS UP
    # --------------------------------------------------------

    if is_thumbs_up(
        landmarks
    ):

        return "THUMBS UP"


    # --------------------------------------------------------
    # OPEN PALM
    # --------------------------------------------------------

    if is_open_palm(
        landmarks
    ):

        return "OPEN PALM"


    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if is_fist(
        landmarks
    ):

        return "FIST"


    # --------------------------------------------------------
    # THREE
    # --------------------------------------------------------

    if is_three_fingers(
        landmarks
    ):

        return "THREE"


    # --------------------------------------------------------
    # TWO
    # --------------------------------------------------------

    if is_two_fingers(
        landmarks
    ):

        return "TWO"


    # --------------------------------------------------------
    # ONE
    # --------------------------------------------------------

    if is_one_finger(
        landmarks
    ):

        return "ONE"


    # --------------------------------------------------------
    # FOUR
    # --------------------------------------------------------

    if is_four_fingers(
        landmarks
    ):

        return "FOUR"


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "UNKNOWN"