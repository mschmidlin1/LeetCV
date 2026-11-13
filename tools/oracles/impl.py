"""Reference implementations used to check the original fixtures and to build extra cases.

These files are not imported by the website and are not shown in the editor.
"""

from __future__ import annotations

import numpy as np
import cv2


class OpenCVError(Exception):
    pass


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_color(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def read_greyscale(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def read_16bit(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def set_region(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[5:15, 5:15] = (0, 255, 0)
    return out


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def save_jpeg(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def _resize_aspect(image: np.ndarray, width: int, interpolation: int) -> np.ndarray:
    height, original_width = image.shape[:2]
    new_height = int(height * width / original_width)
    return cv2.resize(image, (int(width), new_height), interpolation=interpolation)


def resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray:
    return _resize_aspect(image, width, cv2.INTER_LINEAR)


def resize_nearest(image: np.ndarray, width: int) -> np.ndarray:
    return _resize_aspect(image, width, cv2.INTER_NEAREST)


def resize_area(image: np.ndarray, width: int) -> np.ndarray:
    return _resize_aspect(image, width, cv2.INTER_AREA)


def crop_image(image: np.ndarray) -> np.ndarray:
    return image[10:40, 5:35].copy()


def split_and_merge(image: np.ndarray) -> np.ndarray:
    return cv2.merge(cv2.split(image))


def swap_rb(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image)
    return cv2.merge([red, green, blue])


def blue_channel(image: np.ndarray) -> np.ndarray:
    return cv2.split(image)[0]


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) / 255.0


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height))


def rotation_matrix(shape: tuple[int, int], angle: float) -> np.ndarray:
    height, width = shape
    return cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)


def binary_threshold(image: np.ndarray, thresh: int) -> np.ndarray:
    _value, binary = cv2.threshold(gray(image), thresh, 255, cv2.THRESH_BINARY)
    return binary


def binary_inv(image: np.ndarray, thresh: int) -> np.ndarray:
    _value, binary = cv2.threshold(gray(image), thresh, 255, cv2.THRESH_BINARY_INV)
    return binary


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    _value, binary = cv2.threshold(gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(image, (int(ksize), int(ksize)), 0)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def adaptive_mean(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray(image), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def equalize_hist(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray(image))


def clahe(image: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(image))


def mask_hsv(image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kernel)


def _morph_kernel() -> np.ndarray:
    return np.ones((3, 3), np.uint8)


def dilate_image(image: np.ndarray) -> np.ndarray:
    return cv2.dilate(gray(image), _morph_kernel(), iterations=1)


def erode_image(image: np.ndarray) -> np.ndarray:
    return cv2.erode(gray(image), _morph_kernel(), iterations=1)


def morph_open(image: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(gray(image), cv2.MORPH_OPEN, _morph_kernel())


def median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(image, int(ksize))


def canny_edges(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray(image), 100, 200)


def canny(image: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray(image), low, high)


def harris_corners(image: np.ndarray) -> np.ndarray:
    return cv2.cornerHarris(np.float32(gray(image)), 2, 3, 0.04)


def contour_count(image: np.ndarray) -> int:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def contour_all(image: np.ndarray) -> int:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def draw_contours(image: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def sift_keypoint_count(image: np.ndarray) -> int:
    keypoints = cv2.SIFT_create().detect(gray(image), None)
    return 0 if keypoints is None else len(keypoints)


def hough_lines(image: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray(image), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
    if lines is None:
        return np.zeros((0, 2), np.float32)
    rows = lines.reshape(-1, 2).astype(np.float32)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def hough_lines_p(image: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray(image), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=5)
    if lines is None:
        return np.zeros((0, 4), np.int32)
    rows = lines.reshape(-1, 4).astype(np.int32)
    order = np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def orb_match_count(image_a: np.ndarray, image_b: np.ndarray) -> int:
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    _keys_a, desc_a = orb.detectAndCompute(gray(image_a), None)
    _keys_b, desc_b = orb.detectAndCompute(gray(image_b), None)
    if desc_a is None or desc_b is None:
        return 0
    return len(matcher.match(desc_a, desc_b))


def hough_circles(image: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        40,
        param1=80,
        param2=20,
        minRadius=20,
        maxRadius=40,
    )
    if circles is None:
        return np.zeros((0, 3), np.float32)
    return np.asarray(circles[0], dtype=np.float32)


def affine_warp(image: np.ndarray) -> np.ndarray:
    source = np.float32([[10, 10], [50, 10], [10, 50]])
    dest = np.float32([[10, 20], [50, 10], [20, 55]])
    matrix = cv2.getAffineTransform(source, dest)
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height))


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 0)


def flip_both(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, -1)


def perspective_warp(image: np.ndarray) -> np.ndarray:
    source = np.float32([[5, 5], [60, 5], [5, 60], [60, 60]])
    dest = np.float32([[0, 0], [63, 0], [10, 63], [54, 63]])
    matrix = cv2.getPerspectiveTransform(source, dest)
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height))


def shear(image: np.ndarray, shear_factor: float) -> np.ndarray:
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height))


def translate(image: np.ndarray) -> np.ndarray:
    matrix = np.float32([[1, 0, 20], [0, 1, 10]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height))


def scale_image(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(image, matrix, (int(width * sx), int(height * sy)))


def _orb_homography(source: np.ndarray, destination: np.ndarray, query_first: bool) -> np.ndarray | None:
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    keys_a, desc_a = orb.detectAndCompute(gray(source), None)
    keys_b, desc_b = orb.detectAndCompute(gray(destination), None)
    if desc_a is None or desc_b is None:
        return None
    matches = list(matcher.match(desc_a, desc_b))
    if len(matches) < 4:
        return None
    if query_first:
        src_pts = np.float32([keys_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keys_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    else:
        src_pts = np.float32([keys_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keys_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    return matrix


def homography_warp(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    matrix = _orb_homography(source, destination, query_first=True)
    if matrix is None:
        return source
    height, width = destination.shape[:2]
    return cv2.warpPerspective(source, matrix, (width, height))


def farneback_flow(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def stabilize_frame(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    # Map current keypoints to previous keypoints.
    matrix = _orb_homography(previous, current, query_first=False)
    if matrix is None:
        return current
    height, width = previous.shape[:2]
    return cv2.warpPerspective(current, matrix, (width, height))


def read_frames(path: str) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), np.uint8)
    return np.stack(frames, axis=0)


def video_properties(path: str) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return (count, width, height, fps)


def write_video(frames: np.ndarray, path: str) -> None:
    _count, height, width, _channels = frames.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def extract_every_nth(path: str, step: int) -> np.ndarray:
    frames = read_frames(path)
    return frames[::step]


def frame_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return cv2.absdiff(first, second)


def grayscale_video(input_path: str, output_path: str) -> None:
    capture = cv2.VideoCapture(input_path)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        return
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height), False
    )
    writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    writer.release()


def _mog2_mask(frames: np.ndarray) -> np.ndarray:
    subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=16, detectShadows=False)
    mask = None
    for frame in frames:
        mask = subtractor.apply(frame)
    return mask


def moving_boxes(frames: np.ndarray) -> np.ndarray:
    mask = _mog2_mask(frames)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    if not boxes:
        return np.zeros((0, 4), np.int32)
    array = np.asarray(boxes, dtype=np.int32)
    order = np.lexsort((array[:, 1], array[:, 0]))
    return array[order]


def foreground_mask(frames: np.ndarray) -> np.ndarray:
    return _mog2_mask(frames)


def _haar_boxes(image: np.ndarray, cascade_path: str) -> np.ndarray:
    cascade = cv2.CascadeClassifier(cascade_path)
    boxes = cascade.detectMultiScale(gray(image), scaleFactor=1.3, minNeighbors=5)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), np.int32)
    return np.asarray(boxes, dtype=np.int32)


def detect_faces(frames: np.ndarray, cascade_path: str) -> np.ndarray:
    return _haar_boxes(frames[-1], cascade_path)


def haar_faces(image: np.ndarray, cascade_path: str) -> np.ndarray:
    return _haar_boxes(image, cascade_path)


def draw_boxes(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = image.copy()
    for x, y, w, h in np.asarray(boxes):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    return out


def template_top_left(image: np.ndarray, template: np.ndarray) -> tuple[int, int]:
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    return (int(max_loc[0]), int(max_loc[1]))


def template_sqdiff(image: np.ndarray, template: np.ndarray) -> tuple[int, int]:
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    _min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
    return (int(min_loc[0]), int(min_loc[1]))


def template_draw(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    x, y = template_top_left(image, template)
    height, width = template.shape[:2]
    out = image.copy()
    cv2.rectangle(out, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return out


def hog_people(image: np.ndarray) -> np.ndarray:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), np.int32)
    return np.asarray(boxes, dtype=np.int32)


def detect_eyes(image: np.ndarray, cascade_path: str) -> np.ndarray:
    return _haar_boxes(image, cascade_path)


def track_last_box(frames: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "TrackerCSRT_create"):
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = cv2.legacy.TrackerCSRT_create()
    last = (5, 5, 10, 10)
    tracker.init(frames[0], last)
    for frame in frames[1:]:
        ok, box = tracker.update(frame)
        if ok:
            last = box
    return np.asarray(last, dtype=np.float64).reshape(4)


def segment_binary(image: np.ndarray) -> np.ndarray:
    _value, binary = cv2.threshold(gray(image), 200, 255, cv2.THRESH_BINARY)
    return binary


def watershed_segment(image: np.ndarray) -> np.ndarray:
    gray_image = gray(image)
    _value, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)
    distance = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _fg_value, sure_fg = cv2.threshold(distance, 0.5 * float(distance.max()), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    out = image.copy()
    markers = cv2.watershed(out, markers.astype(np.int32))
    out[markers == -1] = (0, 0, 255)
    return out


def _grabcut_labels(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    return mask


def grabcut_foreground(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    labels = _grabcut_labels(image, rect)
    out = image.copy()
    out[~((labels == 1) | (labels == 3))] = 0
    return out


def grabcut_mask(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    labels = _grabcut_labels(image, rect)
    return np.where((labels == 1) | (labels == 3), 255, 0).astype(np.uint8)


def kmeans_segment(image: np.ndarray, k: int) -> np.ndarray:
    cv2.setRNGSeed(0)
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compact, labels, centers = cv2.kmeans(
        pixels, int(k), None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    colored = np.uint8(centers)[labels.flatten()]
    return colored.reshape(image.shape)


def contour_mask(image: np.ndarray) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(out, contours, -1, 255, thickness=-1)
    return out


def watershed_grabcut(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    labels = _grabcut_labels(image, rect)
    markers = np.ones(image.shape[:2], np.int32)
    markers[(labels == 1) | (labels == 3)] = 2
    out = image.copy()
    cv2.watershed(out, markers)
    out[markers == -1] = (0, 0, 255)
    return out


def chessboard_corners(image: np.ndarray) -> np.ndarray:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, (6, 8), flags)
    if not found or corners is None:
        return np.zeros((0, 2), np.float32)
    return np.asarray(corners, dtype=np.float32).reshape(-1, 2)


def camera_matrix(
    object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]
) -> np.ndarray:
    _retval, matrix, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return matrix


def undistort_image(
    image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)
    x, y, roi_width, roi_height = roi
    return undistorted[y : y + roi_height, x : x + roi_width]


def mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _jacobian = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img.reshape(-1, 2), projected.reshape(-1, 2), cv2.NORM_L2)
        errors.append(float(error) / len(np.asarray(obj).reshape(-1, obj.shape[-1])))
    return float(np.mean(errors))


def calibrate_views(
    object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]
) -> np.ndarray:
    return camera_matrix(object_points, image_points, image_size)


def stereo_rotation(
    object_points: list[np.ndarray],
    image_points_left: list[np.ndarray],
    image_points_right: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    result = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        camera_matrix,
        dist_coeffs,
        camera_matrix.copy(),
        dist_coeffs.copy(),
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return result[5]


def stereo_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    return stereo.compute(left, right)


def estimate_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    return rvec, tvec


def prepare_batch(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / np.float32(255)


def sift_descriptors(image: np.ndarray) -> np.ndarray:
    _keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray(image), None)
    return np.asarray(descriptors, dtype=np.float32)


def augment(image: np.ndarray) -> np.ndarray:
    flipped = cv2.flip(image, 1)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, 15], [0, 1, 8]])
    translated = cv2.warpAffine(image, matrix, (width, height))
    return np.stack([flipped, rotated, translated], axis=0)


def classify_patch(features: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> int:
    from sklearn.svm import SVC

    model = SVC(random_state=0)
    model.fit(features, np.asarray(labels).ravel())
    predicted = model.predict(np.asarray(sample, dtype=float).reshape(1, -1))
    return int(predicted[0])


def preprocess_cnn(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / np.float32(255)


def shrink_image(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)


def brighten(image: np.ndarray, amount: int) -> np.ndarray:
    return np.clip(image.astype(np.int16) + int(amount), 0, 255).astype(np.uint8)


def wrap_add(image: np.ndarray, amount: int) -> np.ndarray:
    return np.add(image, np.uint8(amount))


def blur_frames(frames: np.ndarray) -> np.ndarray:
    blurred = [cv2.GaussianBlur(frame, (5, 5), 0) for frame in frames]
    return np.stack(blurred, axis=0)


def blur_chunks(image: np.ndarray, chunk: int) -> np.ndarray:
    parts = []
    width = image.shape[1]
    for start in range(0, width, int(chunk)):
        strip = image[:, start : start + int(chunk)]
        parts.append(cv2.GaussianBlur(strip, (5, 5), 0))
    return np.concatenate(parts, axis=1)


def parallel_orb_counts(images: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create()
    counts = []
    for image in images:
        keypoints = orb.detect(gray(image), None)
        counts.append(0 if keypoints is None else len(keypoints))
    return np.asarray(counts, dtype=np.int32)


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def scale_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float64) * float(factor), 0, 255).astype(np.uint8)


def image_table(paths: list[str]):
    import pandas as pd

    rows = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        rows.append((path, width, height))
    return pd.DataFrame(rows, columns=["path", "width", "height"])


def edge_image(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray(image), 80, 160)


def cluster_pixels(image: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans

    height, width = image.shape[:2]
    pixels = image.reshape(-1, image.shape[-1])
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(pixels)
    return labels.reshape(height, width).astype(np.int32)


def tensorflow_batch(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    scaled = rgb.astype(np.float32) / np.float32(255)
    return scaled[np.newaxis, ...]


def labeled_table(paths: list[str], labels: list[str]):
    import pandas as pd

    rows = []
    for path, label in zip(paths, labels):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        rows.append((path, label, width, height))
    return pd.DataFrame(rows, columns=["path", "label", "width", "height"])


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def bad_resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (0, 0))


def open_video(path: str):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(path)
    return capture


def blur_or_none(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    return cv2.GaussianBlur(image, (5, 5), 0)


def downscale_large(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longer = max(height, width)
    if longer <= max_side:
        return image.copy()
    new_width = int(width * max_side / longer)
    new_height = int(height * max_side / longer)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_frame_with_retry(cap: object, max_attempts: int) -> np.ndarray:
    for _attempt in range(int(max_attempts)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    raise RuntimeError("failed to read frame")


def load_or_raise(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OpenCVError(path)
    return image


def pipeline(path: str) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)


def log_error(path: str, message: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    damaged = np.where(mask != 0, 255, 0).astype(np.uint8)
    return cv2.inpaint(image, damaged, 3, cv2.INPAINT_TELEA)


def inpaint_ns(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    damaged = np.where(mask != 0, 255, 0).astype(np.uint8)
    return cv2.inpaint(image, damaged, 3, cv2.INPAINT_NS)


def channel_histogram(image: np.ndarray, channel: int) -> np.ndarray:
    histogram = cv2.calcHist([image], [int(channel)], None, [256], [0, 256])
    return histogram.reshape(256, 1)


def inpaint_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    height, width = mask.shape
    for x, y in np.asarray(points):
        x = int(x)
        y = int(y)
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        x1 = min(width, x + 2)
        y1 = min(height, y + 2)
        mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def project_origin(
    rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    projected, _jacobian = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return projected.reshape(2).astype(np.float64)


def bgr_histograms(image: np.ndarray) -> np.ndarray:
    histograms = [
        cv2.calcHist([image], [channel], None, [256], [0, 256]).reshape(256) for channel in range(3)
    ]
    return np.stack(histograms, axis=0)


def depth_map(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left, right)
    return disparity.astype(np.float32) / np.float32(16)


def remove_green(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    out = image.copy()
    out[mask > 0] = 0
    return out


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_image = gray(image)
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    return grad_x, grad_y


def laplacian(image: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(gray(image), cv2.CV_32F, ksize=3)


def mosaic(image: np.ndarray, block_size: int) -> np.ndarray:
    out = image.copy()
    height, width = image.shape[:2]
    block = int(block_size)
    for y in range(0, height, block):
        for x in range(0, width, block):
            tile = image[y : y + block, x : x + block]
            color = tile.reshape(-1, tile.shape[-1]).mean(axis=0)
            out[y : y + block, x : x + block] = color.astype(np.uint8)
    return out


def blend(image_a: np.ndarray, image_b: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(image_a, float(alpha), image_b, 1.0 - float(alpha), 0)


def count_objects(image: np.ndarray) -> int:
    contours, _hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for contour in contours if cv2.contourArea(contour) > 50)
