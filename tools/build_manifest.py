"""Build problems/manifest.json from the bank and the interview question list."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "problems" / "bank"
QUESTIONS = Path(r"C:\Users\mschm\source\Computer-Vision-Interview-Preparation\interview question.md")

CONCEPTS = {3, 31, 42, 56, 60, 64, 66, 68, 79, 89, 105, 108, 110, 134}
EXCLUDED = {15, 33, 47, 54, 63, 65, 75, 77, 78, 94, 95, 96, 97, 102, 106, 107, 115, 116, 128, 133, 138}
SLOW = {28, 40, 43, 51, 52, 53, 55, 61, 62, 67, 70, 71, 72, 76, 80, 81, 83, 84, 85, 86, 87, 90, 103}

TOPICS = {
    1: "image basics",
    2: "image basics",
    3: "image basics",
    4: "image basics",
    5: "color",
    6: "color",
    7: "transforms",
    8: "image basics",
    9: "image basics",
    10: "image basics",
    11: "transforms",
    12: "transforms",
    13: "filtering",
    14: "filtering",
    16: "filtering",
    17: "filtering",
    18: "filtering",
    19: "color",
    20: "filtering",
    21: "filtering",
    22: "filtering",
    23: "filtering",
    24: "filtering",
    25: "features",
    26: "features",
    27: "features",
    28: "features",
    29: "features",
    30: "features",
    31: "features",
    32: "features",
    34: "transforms",
    35: "transforms",
    36: "transforms",
    37: "transforms",
    38: "transforms",
    39: "transforms",
    40: "transforms",
    41: "video",
    42: "video",
    43: "video",
    44: "video",
    45: "video",
    46: "video",
    48: "video",
    49: "video",
    50: "video",
    51: "video",
    52: "video",
    53: "detection",
    55: "detection",
    56: "detection",
    57: "detection",
    58: "detection",
    59: "detection",
    60: "detection",
    61: "detection",
    62: "detection",
    64: "detection",
    66: "learning",
    67: "video",
    68: "video",
    69: "segmentation",
    70: "segmentation",
    71: "segmentation",
    72: "segmentation",
    73: "segmentation",
    74: "segmentation",
    76: "segmentation",
    79: "calibration",
    80: "calibration",
    81: "calibration",
    82: "calibration",
    83: "calibration",
    84: "calibration",
    85: "calibration",
    86: "calibration",
    87: "calibration",
    88: "learning",
    89: "learning",
    90: "learning",
    91: "learning",
    92: "learning",
    93: "learning",
    98: "systems",
    99: "systems",
    100: "systems",
    101: "systems",
    103: "systems",
    104: "systems",
    105: "systems",
    108: "systems",
    109: "systems",
    110: "systems",
    111: "systems",
    112: "systems",
    113: "learning",
    114: "learning",
    117: "systems",
    118: "systems",
    119: "systems",
    120: "systems",
    121: "systems",
    122: "systems",
    123: "systems",
    124: "systems",
    125: "systems",
    126: "systems",
    127: "applications",
    129: "applications",
    130: "applications",
    131: "applications",
    132: "applications",
    134: "applications",
    135: "applications",
    136: "applications",
    137: "applications",
    139: "applications",
    140: "applications",
    141: "applications",
}


def prompts() -> dict[int, str]:
    found: dict[int, str] = {}
    current: int | None = None
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+)\.\s+(.*)", line)
        if match:
            current = int(match.group(1))
            found[current] = match.group(2).strip()
        elif current is not None and line.strip() and not line.startswith("#"):
            found[current] += " " + line.strip()
    return found


def title_of(body: str) -> str:
    head = body.split(" Implement `", 1)[0].strip()
    if head.endswith("."):
        head = head[:-1]
    return head


def test_meta(path: Path) -> tuple[str, str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        names = [
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test")
        ]
        vanilla = next(name for name in names if not name.startswith("test_case_"))
        return node.name, vanilla, len(names)
    raise RuntimeError(f"no test class in {path.name}")


QUESTION = re.compile(r"^(what|why|how|explain|compare|what's)\b", re.I)

# Coding-task titles for prompts whose opening sentence is a concept question.
CODE_TITLES = {
    1: "Load an image",
    2: "Set a region of pixel values",
    4: "Save an image",
    5: "Convert an image to grayscale",
    6: "Convert an image from BGR to HSV",
    8: "Crop an image",
    9: "Split and merge image channels",
    12: "Build a rotation matrix",
    13: "Threshold an image to binary",
    14: "Apply a Gaussian blur",
    16: "Apply an adaptive threshold",
    18: "Equalize a histogram",
    19: "Mask pixels by HSV range",
    22: "Reduce noise with a median blur",
    23: "Detect edges with fixed Canny thresholds",
    24: "Detect edges with chosen Canny thresholds",
    25: "Detect corners with Harris",
    26: "Count external contours",
    28: "Count SIFT keypoints",
    29: "Detect lines with the Hough transform",
    34: "Warp an image with an affine transform",
    35: "Flip an image horizontally",
    36: "Warp an image with a perspective transform",
    38: "Translate an image",
    41: "Compute Farneback optical flow",
    44: "Read every frame from a video",
    45: "Read frame count, size, and fps from a video",
    46: "Write frames to a video file",
    49: "Subtract consecutive frames",
    52: "Build a MOG2 foreground mask",
    55: "Detect faces with a Haar cascade",
    57: "Draw bounding boxes",
    58: "Find a template match",
    61: "Detect pedestrians with HOG",
    69: "Segment an image with a fixed threshold",
    70: "Segment an image with watershed",
    71: "Extract a GrabCut foreground",
    72: "Build a GrabCut mask",
    74: "Fill external contours into a mask",
    80: "Find chessboard corners",
    81: "Calibrate a camera and return the camera matrix",
    83: "Compute mean reprojection error",
    85: "Calibrate a stereo pair and return the rotation",
    88: "Prepare an image for a model",
    90: "Extract SIFT descriptors",
    92: "Classify a feature vector with an SVM",
    98: "Shrink an image to 32 by 32",
    99: "Add a constant to every pixel",
    100: "Blur every frame of a stack",
    104: "Resize a frame",
    109: "Scale pixel brightness",
    111: "Build a table of image sizes",
    113: "Cluster pixels with k-means",
    118: "Load an image or raise FileNotFoundError",
    119: "Resize to an invalid size",
    120: "Open a video or raise RuntimeError",
    122: "Downscale an image that is too large",
    126: "Append an error message to a log file",
    127: "Inpaint damaged pixels",
    129: "Compute one channel histogram",
    131: "Project a 3D point into the image",
}

TITLE_OVERRIDES = {
    1: "Load color, grayscale, and 16-bit images",
    4: "Save a PNG and a JPEG",
    5: "Convert BGR to grayscale and to RGB",
    6: "Convert between BGR and HSV",
    7: "Resize with linear, nearest, and area interpolation",
    9: "Split channels, swap red and blue, and return blue",
    13: "Threshold with binary, inverse, and Otsu",
    16: "Apply Gaussian and mean adaptive thresholds",
    18: "Equalize a histogram and apply CLAHE",
    21: "Dilate, erode, and open",
    26: "Count external contours and every contour",
    29: "Detect lines with standard and probabilistic Hough",
    35: "Flip an image horizontally, vertically, and both",
    58: "Locate a template with correlation and squared difference",
    99: "Add a constant with saturation and with wraparound",
    127: "Inpaint with Telea and Navier-Stokes",
    137: "Compute Sobel gradients and the Laplacian",
}

PROMPT_OVERRIDES = {
    1: "Implement `read_color(path: str) -> np.ndarray`, `read_greyscale(path: str) -> np.ndarray`, and `read_16bit(path: str) -> np.ndarray`. `read_color` returns an 8-bit BGR image. `read_greyscale` returns an 8-bit single-channel image. `read_16bit` returns a 16-bit grayscale file as a single-channel `uint16` image, without scaling it down to 8 bits.",
    4: "Implement `save_image(image: np.ndarray, path: str) -> None` and `save_jpeg(image: np.ndarray, path: str) -> None`. `save_image` writes `image` to `path` with OpenCV's default parameters. The PNG tests use a `.png` path. `save_jpeg` writes a JPEG at quality 90.",
    5: "Implement `to_gray(image: np.ndarray) -> np.ndarray` and `to_rgb(image: np.ndarray) -> np.ndarray`. Treat `image` as BGR. `to_gray` returns a grayscale `uint8` image. `to_rgb` returns an RGB `uint8` image.",
    6: "Implement `to_hsv(image: np.ndarray) -> np.ndarray` and `hsv_to_bgr(image: np.ndarray) -> np.ndarray`. `to_hsv` treats `image` as BGR and returns HSV `uint8`, with hue in OpenCV's 0-179 range. `hsv_to_bgr` treats `image` as that HSV image and returns BGR `uint8`.",
    7: "Implement `resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray`, `resize_nearest(image: np.ndarray, width: int) -> np.ndarray`, and `resize_area(image: np.ndarray, width: int) -> np.ndarray`. For each, the new height is `int(height * width / original_width)`. Use linear, nearest-neighbor, and area interpolation respectively.",
    9: "Implement `split_and_merge(image: np.ndarray) -> np.ndarray`, `swap_rb(image: np.ndarray) -> np.ndarray`, and `blue_channel(image: np.ndarray) -> np.ndarray`. Split the BGR channels. `split_and_merge` merges them in the original order. `swap_rb` merges them as red, green, blue. `blue_channel` returns the blue channel.",
    13: "Implement `binary_threshold(image: np.ndarray, thresh: int) -> np.ndarray`, `binary_inv(image: np.ndarray, thresh: int) -> np.ndarray`, and `otsu_threshold(image: np.ndarray) -> np.ndarray`. Convert to grayscale. The first two use a fixed threshold of `thresh` and a maximum of 255, with the normal and inverse binary rules. `otsu_threshold` uses Otsu's threshold and the normal binary rule, maximum 255.",
    16: "Implement `adaptive_threshold(image: np.ndarray) -> np.ndarray` and `adaptive_mean(image: np.ndarray) -> np.ndarray`. Convert to grayscale. Both use the binary type, block size 11, `C` of 2, and a maximum of 255. `adaptive_threshold` uses a Gaussian weighted mean. `adaptive_mean` uses the local mean.",
    18: "Implement `equalize_hist(image: np.ndarray) -> np.ndarray` and `clahe(image: np.ndarray) -> np.ndarray`. Convert to grayscale. `equalize_hist` equalizes the whole histogram. `clahe` uses a clip limit of 2 and 8 by 8 tiles.",
    21: "Implement `dilate_image(image: np.ndarray) -> np.ndarray`, `erode_image(image: np.ndarray) -> np.ndarray`, and `morph_open(image: np.ndarray) -> np.ndarray`. Convert to grayscale and use one pass of a 3x3 kernel of ones. Dilate, erode, or open.",
    26: "Implement `contour_count(image: np.ndarray) -> int` and `contour_all(image: np.ndarray) -> int`. `image` is a single-channel `uint8` mask. Use simple chain approximation. `contour_count` returns how many external contours it has. `contour_all` returns how many contours it has, including holes.",
    29: "Implement `hough_lines(image: np.ndarray) -> np.ndarray` and `hough_lines_p(image: np.ndarray) -> np.ndarray`. Canny thresholds are 50 and 150. Both use rho 1, theta `pi / 180`, and threshold 30. `hough_lines` returns `float32` rows `(rho, theta)` sorted by rho, then theta, or shape `(0, 2)` when there are no lines. `hough_lines_p` uses a minimum line length of 20 and a maximum gap of 5, and returns `int32` rows `(x1, y1, x2, y2)` sorted by x1, y1, x2, then y2, or shape `(0, 4)` when there are none.",
    35: "Implement `flip_horizontal(image: np.ndarray) -> np.ndarray`, `flip_vertical(image: np.ndarray) -> np.ndarray`, and `flip_both(image: np.ndarray) -> np.ndarray`. Flip across the vertical axis, the horizontal axis, or both.",
    58: "Implement `template_top_left(image: np.ndarray, template: np.ndarray) -> tuple[int, int]` and `template_sqdiff(image: np.ndarray, template: np.ndarray) -> tuple[int, int]`. Both return the top-left corner of the best match. `template_top_left` uses normalized correlation, whose best match is the maximum. `template_sqdiff` uses squared difference, whose best match is the minimum.",
    99: "Implement `brighten(image: np.ndarray, amount: int) -> np.ndarray` and `wrap_add(image: np.ndarray, amount: int) -> np.ndarray`. Both add `amount` to every pixel and return `uint8`. `brighten` clips to 0..255. `wrap_add` wraps around, as `uint8` addition does.",
    127: "Implement `inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray` and `inpaint_ns(image: np.ndarray, mask: np.ndarray) -> np.ndarray`. `mask` is `uint8`, with nonzero damaged pixels. Both use radius 3. `inpaint` uses Telea. `inpaint_ns` uses Navier-Stokes.",
    137: "Implement `sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]` and `laplacian(image: np.ndarray) -> np.ndarray`. Convert to grayscale. `sobel_gradients` returns `(grad_x, grad_y)` as `float32` Sobel gradients with ksize 3. `laplacian` returns the `float32` Laplacian with ksize 3.",
}

TASK_TITLES = {
    21: "Dilate a grayscale image",
    86: "Compute stereo disparity",
    112: "Compute Canny edges",
    121: "Blur an image, or return None",
    132: "Compute a histogram for each BGR channel",
}


def concept_text(lead: str) -> str:
    text = lead.strip()
    text = re.sub(
        r"\s+(?:and\s+)?(?:provide|give) an example(?: of [^?.]+)?\.?$",
        "",
        text,
        flags=re.I,
    ).strip()
    if text.endswith("."):
        text = text[:-1]
    return text


def task_title(lead: str) -> str:
    text = lead.strip().rstrip(".")
    for prefix in (
        "Write a function to ",
        "Write a function with ",
        "Write a function for ",
        "Implement a function to ",
        "Implement a function for ",
        "Implement a function ",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
            break
    return text[:1].upper() + text[1:]


def concept_entry(problem: dict, concept: str) -> dict:
    return {
        "id": f"{problem['id']}c",
        "number": problem["number"],
        "title": concept,
        "prompt": concept,
        "topic": problem["topic"],
        "timeout": 30,
        "type": "critical_thinking",
        "starter": "",
        "test_module": "",
        "test_class": "",
        "vanilla_test": "",
        "solution_file": "",
        "test_count": 0,
    }


def polish(problems: list[dict]) -> list[dict]:
    """Pull concept questions out of coding prompts. Keep the graded spec on the code problem."""
    existing = {
        item["title"].casefold()
        for item in problems
        if item["type"] == "critical_thinking"
    }
    polished: list[dict] = []
    for problem in problems:
        if problem["type"] != "code" or " Implement `" not in problem["prompt"]:
            polished.append(problem)
            continue
        lead, spec = problem["prompt"].split(" Implement `", 1)
        spec = "Implement `" + spec
        if QUESTION.match(lead):
            concept = concept_text(lead)
            if concept.casefold() not in existing:
                polished.append(concept_entry(problem, concept))
                existing.add(concept.casefold())
            number = problem["number"]
            if number not in CODE_TITLES:
                raise SystemExit(f"missing code title for question {number}")
            problem["title"] = CODE_TITLES[number]
        else:
            number = problem["number"]
            problem["title"] = TASK_TITLES.get(number, task_title(lead))
        problem["prompt"] = PROMPT_OVERRIDES.get(problem["number"], spec)
        if problem["number"] in TITLE_OVERRIDES:
            problem["title"] = TITLE_OVERRIDES[problem["number"]]
        if QUESTION.match(problem["title"]):
            raise SystemExit(f"code title is still a question: {problem['id']}")
        polished.append(problem)
    return polished


def main() -> None:
    texts = prompts()
    problems = []
    for number in range(1, 142):
        if number in EXCLUDED:
            continue
        if number not in TOPICS:
            raise SystemExit(f"missing topic for {number}")
        body = texts[number]
        problem_id = f"q{number:03d}"
        entry = {
            "id": problem_id,
            "number": number,
            "title": title_of(body),
            "prompt": body,
            "topic": TOPICS[number],
            "timeout": 120 if number in SLOW else 30,
        }
        if number in CONCEPTS:
            entry.update(
                {
                    "type": "critical_thinking",
                    "starter": "",
                    "test_module": "",
                    "test_class": "",
                    "vanilla_test": "",
                    "solution_file": "",
                    "test_count": 0,
                }
            )
        else:
            solution = next((BANK / "solutions").glob(f"{problem_id}_*.py"))
            test = next((BANK / "tests").glob(f"test_{problem_id}_*.py"))
            test_class, vanilla, count = test_meta(test)
            entry.update(
                {
                    "type": "code",
                    "starter": solution.read_text(encoding="utf-8"),
                    "test_module": f"tests.{test.stem}",
                    "test_class": test_class,
                    "vanilla_test": vanilla,
                    "solution_file": f"solutions/{solution.name}",
                    "test_count": count,
                }
            )
        problems.append(entry)
    problems = polish(problems)
    code = sum(1 for item in problems if item["type"] == "code")
    concepts = sum(1 for item in problems if item["type"] == "critical_thinking")
    if code != 106 or concepts < 14:
        raise SystemExit(f"unexpected counts code={code} concepts={concepts}")
    destination = ROOT / "problems" / "manifest.json"
    destination.write_text(json.dumps(problems, indent=2), encoding="utf-8")
    print(f"wrote {len(problems)} problems to {destination}")


if __name__ == "__main__":
    main()
