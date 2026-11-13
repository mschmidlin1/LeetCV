"""Add two extra cases to every judged problem and check them with the oracles."""

from __future__ import annotations

import json
import re
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np

from oracles import impl

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "problems" / "bank"
TESTS = BANK / "tests"
CASES = BANK / "fixtures" / "cases"

SKIP_MARKERS = ("tempfile", "assertRaises", "read_bytes", "FakeCap", "columns", "clip.avi")


def image_like(array: np.ndarray) -> bool:
    array = np.asarray(array)
    if array.ndim == 4:
        return array.shape[1] >= 8 and array.shape[2] >= 8
    if array.ndim >= 2:
        return array.shape[0] >= 8 and array.shape[1] >= 8
    return False


def flip_width(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    axis = 2 if array.ndim == 4 else 1
    return np.ascontiguousarray(np.flip(array, axis=axis))


def paint_value(array: np.ndarray):
    if np.issubdtype(array.dtype, np.floating) and float(np.max(array)) <= 1.5:
        return array.dtype.type(0.9)
    return np.array(220, dtype=array.dtype)


def paint(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    value = paint_value(out)
    if out.ndim == 4:
        out[:, 1:12, 1:12, ...] = value
    else:
        out[1:12, 1:12, ...] = value
    return np.ascontiguousarray(out)


def invert(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.bitwise_not(array)
    flipped = flip_width(array)
    if flipped.dtype == array.dtype and flipped.shape == array.shape:
        return np.ascontiguousarray(np.flip(array, axis=0))
    return flipped


def copy_inputs(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in base.items()}


def nudge(array: np.ndarray, delta: float) -> np.ndarray:
    updated = np.array(array, copy=True)
    if np.issubdtype(updated.dtype, np.floating):
        updated.flat[0] = updated.flat[0] + updated.dtype.type(delta)
    else:
        updated.flat[0] = updated.flat[0] + 1
    return updated


def candidates(base: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    options = []
    for mutate in (flip_width, paint, invert):
        updated = copy_inputs(base)
        changed = False
        for key, value in base.items():
            if image_like(value):
                updated[key] = mutate(value)
                changed = True
        if changed and not same_inputs(updated, base):
            options.append(updated)
    if options:
        return options
    largest = max(base, key=lambda key: base[key].size)
    for delta in (0.25, -0.5, 1.0):
        updated = copy_inputs(base)
        updated[largest] = nudge(base[largest], delta)
        options.append(updated)
    return options


def same_inputs(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(np.array_equal(left[key], right[key]) for key in left)


def parse_method(text: str) -> dict | None:
    method = re.search(r"    def test_\w+\(self\):.*?(?=\n    def |\nif __name__|\Z)", text, re.S)
    if method is None:
        return None
    text = method.group(0)
    if any(marker in text for marker in SKIP_MARKERS):
        return None
    loads = [
        (name, filename)
        for name, filename in re.findall(r"(\w+) = np\.load\(CASE / '([^']+\.npy)'\)", text)
        if name != "expected"
    ]
    call = re.search(r"^\s*actual = (.+)$", text, re.M)
    if not loads or not call:
        return None
    if "expected.npz" in text and "first, second = actual" in text:
        kind = "npz"
    elif "expected.json" in text:
        kind = "json"
    elif "assert_allclose" in text:
        kind = "close"
    else:
        kind = "equal"
    rtol = "1e-05"
    atol = "0.0001"
    rtol_match = re.search(r"rtol=([^,\n\)]+)", text)
    atol_match = re.search(r"atol=([^,\n\)]+)", text)
    if rtol_match:
        rtol = rtol_match.group(1).strip()
    if atol_match:
        atol = atol_match.group(1).strip()
    return {"loads": loads, "call": call.group(1).strip(), "kind": kind, "rtol": rtol, "atol": atol}


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_expected(path: Path, value, kind: str) -> None:
    if kind in {"equal", "close"}:
        np.save(path, np.asarray(value))
    elif kind == "json":
        path.write_text(json.dumps(to_jsonable(value)), encoding="utf-8")
    elif kind == "npz":
        first, second = value
        np.savez(path, first=np.asarray(first), second=np.asarray(second))
    else:
        raise ValueError(kind)


def assertion_block(kind: str, filename: str, rtol: str, atol: str) -> str:
    if kind == "equal":
        return (
            f"        expected = np.load(CASE / '{filename}')\n"
            "        np.testing.assert_array_equal(actual, expected)\n"
        )
    if kind == "close":
        return (
            f"        expected = np.load(CASE / '{filename}')\n"
            f"        np.testing.assert_allclose(actual, expected, rtol={rtol}, atol={atol})\n"
        )
    if kind == "json":
        return (
            f"        expected = json.loads((CASE / '{filename}').read_text(encoding='utf-8'))\n"
            "        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))\n"
        )
    if kind == "npz":
        return (
            f"        expected = np.load(CASE / '{filename}')\n"
            "        first, second = actual\n"
            f"        np.testing.assert_allclose(first, expected['first'], rtol={rtol}, atol={atol})\n"
            f"        np.testing.assert_allclose(second, expected['second'], rtol={rtol}, atol={atol})\n"
        )
    raise ValueError(kind)


def method_source(name: str, loads: list[tuple[str, str]], call: str, kind: str, expected_name: str, rtol: str, atol: str) -> str:
    lines = [f"    def {name}(self):"]
    for variable, filename in loads:
        lines.append(f"        {variable} = np.load(CASE / '{filename}')")
    lines.append(f"        actual = {call}")
    lines.append(assertion_block(kind, expected_name, rtol, atol).rstrip("\n"))
    return "\n".join(lines) + "\n"


def ensure_import(text: str, statement: str) -> str:
    if statement in text:
        return text
    return statement + "\n" + text


def append_methods(path: Path, methods: str, imports: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for statement in imports:
        text = ensure_import(text, statement)
    needle = '\nif __name__ == "__main__":'
    if needle not in text:
        raise RuntimeError(f"no main guard in {path.name}")
    path.write_text(text.replace(needle, "\n" + methods + needle, 1), encoding="utf-8")


def run_call(function_name: str, call: str, arrays: dict[str, np.ndarray]):
    namespace = {function_name: getattr(impl, function_name), "np": np, "cv2": cv2, "ROOT": BANK, "Path": Path}
    namespace.update(arrays)
    return eval(call, {"np": np, "cv2": cv2, "ROOT": BANK, "Path": Path}, namespace)  # noqa: S307 - call text is our own tests


def case_dir_for(question: str) -> Path:
    return CASES / question.split("_", 1)[0]


def extend_simple(path: Path, function_name: str, parsed: dict) -> str:
    case_dir = case_dir_for(path.stem.removeprefix("test_"))
    base = {}
    for variable, filename in parsed["loads"]:
        base[variable] = np.load(case_dir / filename)
    chosen = []
    for option in candidates(base):
        try:
            value = run_call(function_name, parsed["call"], option)
        except Exception:
            continue
        kind = parsed["kind"]
        if kind == "equal" and not isinstance(value, np.ndarray):
            kind = "json"
        if kind == "close" and not isinstance(value, np.ndarray):
            kind = "json"
        if kind == "npz" and not (isinstance(value, tuple) and len(value) == 2):
            continue
        chosen.append((option, value, kind))
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return f"short:{len(chosen)}"
    methods = []
    imports = []
    for index, (arrays, value, kind) in enumerate(chosen):
        label = "b" if index == 0 else "c"
        loads = []
        for variable, _filename in parsed["loads"]:
            filename = f"case_{label}_{variable}.npy"
            np.save(case_dir / filename, arrays[variable])
            loads.append((variable, filename))
        expected_name = f"case_{label}_expected.npz" if kind == "npz" else (
            f"case_{label}_expected.json" if kind == "json" else f"case_{label}_expected.npy"
        )
        save_expected(case_dir / expected_name, value, kind)
        if kind == "json":
            imports.append("import json")
        methods.append(
            method_source(
                f"test_case_{label}",
                loads,
                parsed["call"],
                kind,
                expected_name,
                parsed["rtol"],
                parsed["atol"],
            )
        )
    append_methods(path, "\n".join(methods) + "\n", imports)
    return "simple"


def write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def write_clip(path: Path, frames: np.ndarray) -> None:
    impl.write_video(frames, str(path))


def solid_clip(count: int, height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    frames = np.zeros((count, height, width, 3), np.uint8)
    frames[:] = color
    return frames


def load_list(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[key] for key in sorted(data.files)]


def save_list(path: Path, arrays: list[np.ndarray]) -> None:
    payload = {f"a{index:02d}": np.asarray(array) for index, array in enumerate(arrays)}
    np.savez(path, **payload)


def shift_points(arrays: list[np.ndarray], delta: float) -> list[np.ndarray]:
    shifted = []
    for array in arrays:
        updated = np.array(array, dtype=np.float32, copy=True)
        updated[..., 0] += np.float32(delta)
        shifted.append(updated)
    return shifted


def pattern(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = color
    cv2.rectangle(image, (2, 2), (width // 2, height // 2), (0, 255, 0), -1)
    return image


def extend_special(path: Path, question: str) -> str:
    case_dir = case_dir_for(question)
    function = question.split("_", 1)[1]
    methods: list[str] = []
    imports: list[str] = []

    if question == "q001_read_image":
        for label, image in (("b", pattern(20, 30, (10, 20, 30))), ("c", pattern(18, 18, (200, 10, 10)))):
            png = case_dir / f"case_{label}.png"
            write_png(png, image)
            np.save(case_dir / f"case_{label}_expected.npy", impl.read_color(str(png)))
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        actual = read_color(str(CASE / 'case_{label}.png'))
                        expected = np.load(CASE / 'case_{label}_expected.npy')
                        np.testing.assert_array_equal(actual, expected)
                    """
                )
            )
    elif question == "q004_save_image":
        for label, image in (("b", pattern(12, 16, (4, 5, 6))), ("c", pattern(9, 11, (255, 128, 0)))):
            np.save(case_dir / f"case_{label}_image.npy", image)
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        import cv2
                        image = np.load(CASE / 'case_{label}_image.npy')
                        folder = tempfile.mkdtemp()
                        path = str(Path(folder) / 'out.png')
                        save_image(image, path)
                        np.testing.assert_array_equal(cv2.imread(path), image)
                    """
                )
            )
    elif question == "q012_rotation_matrix":
        for label, shape, angle in (("b", (32, 48), 90.0), ("c", (20, 20), -15.0)):
            np.save(case_dir / f"case_{label}_expected.npy", impl.rotation_matrix(shape, angle))
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        actual = rotation_matrix({shape}, {angle})
                        expected = np.load(CASE / 'case_{label}_expected.npy')
                        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)
                    """
                )
            )
    elif question in {"q044_read_frames", "q045_video_properties", "q048_extract_every_nth"}:
        clips = {
            "b": solid_clip(2, 16, 24, (0, 0, 255)),
            "c": solid_clip(4, 12, 18, (0, 255, 0)),
        }
        for label, frames in clips.items():
            avi = case_dir / f"case_{label}.avi"
            write_clip(avi, frames)
            if question == "q044_read_frames":
                np.save(case_dir / f"case_{label}_expected.npy", impl.read_frames(str(avi)))
                methods.append(
                    textwrap.dedent(
                        f"""\
                        def test_case_{label}(self):
                            actual = read_frames(str(CASE / 'case_{label}.avi'))
                            expected = np.load(CASE / 'case_{label}_expected.npy')
                            np.testing.assert_array_equal(actual, expected)
                        """
                    )
                )
            elif question == "q045_video_properties":
                (case_dir / f"case_{label}_expected.json").write_text(
                    json.dumps(list(impl.video_properties(str(avi)))), encoding="utf-8"
                )
                imports.append("import json")
                methods.append(
                    textwrap.dedent(
                        f"""\
                        def test_case_{label}(self):
                            actual = video_properties(str(CASE / 'case_{label}.avi'))
                            expected = json.loads((CASE / 'case_{label}_expected.json').read_text(encoding='utf-8'))
                            self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))
                        """
                    )
                )
            else:
                step = 1 if label == "b" else 2
                np.save(case_dir / f"case_{label}_expected.npy", impl.extract_every_nth(str(avi), step))
                methods.append(
                    textwrap.dedent(
                        f"""\
                        def test_case_{label}(self):
                            actual = extract_every_nth(str(CASE / 'case_{label}.avi'), {step})
                            expected = np.load(CASE / 'case_{label}_expected.npy')
                            np.testing.assert_array_equal(actual, expected)
                        """
                    )
                )
    elif question == "q046_write_video":
        for label, frames in (
            ("b", solid_clip(2, 16, 24, (30, 40, 50))),
            ("c", solid_clip(3, 12, 20, (8, 9, 10))),
        ):
            np.save(case_dir / f"case_{label}_frames.npy", frames)
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        import cv2
                        frames = np.load(CASE / 'case_{label}_frames.npy')
                        folder = tempfile.mkdtemp()
                        path = str(Path(folder) / 'out.avi')
                        write_video(frames, path)
                        capture = cv2.VideoCapture(path)
                        decoded = []
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            decoded.append(frame)
                        capture.release()
                        self.assertEqual(len(decoded), frames.shape[0])
                        self.assertEqual(decoded[0].shape[0], frames.shape[1])
                        self.assertEqual(decoded[0].shape[1], frames.shape[2])
                    """
                )
            )
    elif question == "q050_grayscale_video":
        for label, frames in (
            ("b", solid_clip(2, 16, 24, (20, 80, 200))),
            ("c", solid_clip(3, 12, 18, (255, 0, 0))),
        ):
            avi = case_dir / f"case_{label}.avi"
            write_clip(avi, frames)
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        import cv2
                        folder = tempfile.mkdtemp()
                        output_path = str(Path(folder) / 'out.avi')
                        grayscale_video(str(CASE / 'case_{label}.avi'), output_path)
                        capture = cv2.VideoCapture(output_path)
                        decoded = []
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            decoded.append(frame)
                        capture.release()
                        self.assertEqual(len(decoded), {frames.shape[0]})
                        self.assertEqual(decoded[0].shape[0], {frames.shape[1]})
                        self.assertEqual(decoded[0].shape[1], {frames.shape[2]})
                    """
                )
            )
    elif question in {"q081_camera_matrix", "q084_calibrate_views"}:
        fn = impl.camera_matrix if question.startswith("q081") else impl.calibrate_views
        name = "camera_matrix" if question.startswith("q081") else "calibrate_views"
        for label, delta in (("b", 0.35), ("c", -0.2)):
            objects = load_list(case_dir / "object_points.npz")
            images = shift_points(load_list(case_dir / "image_points.npz"), delta)
            expected = fn(objects, images, (160, 120))
            save_list(case_dir / f"case_{label}_object_points.npz", objects)
            save_list(case_dir / f"case_{label}_image_points.npz", images)
            np.save(case_dir / f"case_{label}_expected.npy", expected)
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        _object_points = np.load(CASE / 'case_{label}_object_points.npz')
                        object_points = [_object_points[key] for key in sorted(_object_points.files)]
                        _image_points = np.load(CASE / 'case_{label}_image_points.npz')
                        image_points = [_image_points[key] for key in sorted(_image_points.files)]
                        actual = {name}(object_points, image_points, (160, 120))
                        expected = np.load(CASE / 'case_{label}_expected.npy')
                        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)
                    """
                )
            )
    elif question == "q083_mean_reprojection_error":
        objects = load_list(case_dir / "object_points.npz")
        rvecs = load_list(case_dir / "rvecs.npz")
        tvecs = load_list(case_dir / "tvecs.npz")
        camera = np.load(case_dir / "camera_matrix.npy")
        dist = np.load(case_dir / "dist_coeffs.npy")
        for label, delta in (("b", 0.5), ("c", -0.25)):
            images = shift_points(load_list(case_dir / "image_points.npz"), delta)
            expected = impl.mean_reprojection_error(objects, images, rvecs, tvecs, camera, dist)
            save_list(case_dir / f"case_{label}_object_points.npz", objects)
            save_list(case_dir / f"case_{label}_image_points.npz", images)
            save_list(case_dir / f"case_{label}_rvecs.npz", rvecs)
            save_list(case_dir / f"case_{label}_tvecs.npz", tvecs)
            np.save(case_dir / f"case_{label}_camera_matrix.npy", camera)
            np.save(case_dir / f"case_{label}_dist_coeffs.npy", dist)
            np.save(case_dir / f"case_{label}_expected.npy", np.array(expected))
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        _object_points = np.load(CASE / 'case_{label}_object_points.npz')
                        object_points = [_object_points[key] for key in sorted(_object_points.files)]
                        _image_points = np.load(CASE / 'case_{label}_image_points.npz')
                        image_points = [_image_points[key] for key in sorted(_image_points.files)]
                        _rvecs = np.load(CASE / 'case_{label}_rvecs.npz')
                        rvecs = [_rvecs[key] for key in sorted(_rvecs.files)]
                        _tvecs = np.load(CASE / 'case_{label}_tvecs.npz')
                        tvecs = [_tvecs[key] for key in sorted(_tvecs.files)]
                        camera_matrix = np.load(CASE / 'case_{label}_camera_matrix.npy')
                        dist_coeffs = np.load(CASE / 'case_{label}_dist_coeffs.npy')
                        actual = mean_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
                        expected = np.load(CASE / 'case_{label}_expected.npy')
                        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)
                    """
                )
            )
    elif question == "q085_stereo_rotation":
        objects = load_list(case_dir / "object_points.npz")
        camera = np.load(case_dir / "camera_matrix.npy")
        dist = np.load(case_dir / "dist_coeffs.npy")
        for label, delta in (("b", 0.3), ("c", -0.15)):
            left = shift_points(load_list(case_dir / "image_points_left.npz"), delta)
            right = shift_points(load_list(case_dir / "image_points_right.npz"), -delta)
            expected = impl.stereo_rotation(objects, left, right, camera, dist, (160, 120))
            save_list(case_dir / f"case_{label}_object_points.npz", objects)
            save_list(case_dir / f"case_{label}_image_points_left.npz", left)
            save_list(case_dir / f"case_{label}_image_points_right.npz", right)
            np.save(case_dir / f"case_{label}_camera_matrix.npy", camera)
            np.save(case_dir / f"case_{label}_dist_coeffs.npy", dist)
            np.save(case_dir / f"case_{label}_expected.npy", expected)
            methods.append(
                textwrap.dedent(
                    f"""\
                    def test_case_{label}(self):
                        _object_points = np.load(CASE / 'case_{label}_object_points.npz')
                        object_points = [_object_points[key] for key in sorted(_object_points.files)]
                        _image_points_left = np.load(CASE / 'case_{label}_image_points_left.npz')
                        image_points_left = [_image_points_left[key] for key in sorted(_image_points_left.files)]
                        _image_points_right = np.load(CASE / 'case_{label}_image_points_right.npz')
                        image_points_right = [_image_points_right[key] for key in sorted(_image_points_right.files)]
                        camera_matrix = np.load(CASE / 'case_{label}_camera_matrix.npy')
                        dist_coeffs = np.load(CASE / 'case_{label}_dist_coeffs.npy')
                        actual = stereo_rotation(object_points, image_points_left, image_points_right, camera_matrix, dist_coeffs, (160, 120))
                        expected = np.load(CASE / 'case_{label}_expected.npy')
                        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)
                    """
                )
            )
    elif question == "q111_image_table":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    import cv2
                    folder = tempfile.mkdtemp()
                    path = str(Path(folder) / 'pic.png')
                    cv2.imwrite(path, np.zeros((15, 21, 3), np.uint8))
                    actual = image_table([path])
                    self.assertEqual(list(actual.columns), ["path", "width", "height"])
                    self.assertEqual(actual["path"].tolist(), [path])
                    self.assertEqual(actual["width"].tolist(), [21])
                    self.assertEqual(actual["height"].tolist(), [15])

                def test_case_c(self):
                    import cv2
                    folder = tempfile.mkdtemp()
                    first = str(Path(folder) / 'a.png')
                    second = str(Path(folder) / 'b.png')
                    cv2.imwrite(first, np.zeros((9, 13, 3), np.uint8))
                    cv2.imwrite(second, np.zeros((30, 10, 3), np.uint8))
                    actual = image_table([first, second])
                    self.assertEqual(actual["width"].tolist(), [13, 10])
                    self.assertEqual(actual["height"].tolist(), [9, 30])
                """
            )
        )
    elif question == "q117_labeled_table":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    import cv2
                    folder = tempfile.mkdtemp()
                    path = str(Path(folder) / 'pic.png')
                    cv2.imwrite(path, np.zeros((14, 19, 3), np.uint8))
                    actual = labeled_table([path], ["cat"])
                    self.assertEqual(list(actual.columns), ["path", "label", "width", "height"])
                    self.assertEqual(actual["label"].tolist(), ["cat"])
                    self.assertEqual(actual["width"].tolist(), [19])
                    self.assertEqual(actual["height"].tolist(), [14])

                def test_case_c(self):
                    import cv2
                    folder = tempfile.mkdtemp()
                    first = str(Path(folder) / 'a.png')
                    second = str(Path(folder) / 'b.png')
                    cv2.imwrite(first, np.zeros((8, 12, 3), np.uint8))
                    cv2.imwrite(second, np.zeros((22, 7, 3), np.uint8))
                    actual = labeled_table([first, second], ["one", "two"])
                    self.assertEqual(actual["label"].tolist(), ["one", "two"])
                    self.assertEqual(actual["width"].tolist(), [12, 7])
                    self.assertEqual(actual["height"].tolist(), [8, 22])
                """
            )
        )
    elif question == "q118_load_image":
        image = pattern(17, 23, (1, 2, 3))
        write_png(case_dir / "case_b.png", image)
        np.save(case_dir / "case_b_expected.npy", impl.load_image(str(case_dir / "case_b.png")))
        image_c = pattern(13, 13, (9, 8, 7))
        write_png(case_dir / "case_c.png", image_c)
        np.save(case_dir / "case_c_expected.npy", impl.load_image(str(case_dir / "case_c.png")))
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    actual = load_image(str(CASE / 'case_b.png'))
                    np.testing.assert_array_equal(actual, np.load(CASE / 'case_b_expected.npy'))
                    with self.assertRaises(FileNotFoundError):
                        load_image(str(CASE / 'missing-b.png'))

                def test_case_c(self):
                    actual = load_image(str(CASE / 'case_c.png'))
                    np.testing.assert_array_equal(actual, np.load(CASE / 'case_c_expected.npy'))
                """
            )
        )
    elif question == "q119_bad_resize":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    import cv2
                    image = np.zeros((8, 9, 3), np.uint8)
                    with self.assertRaises(cv2.error):
                        bad_resize(image)

                def test_case_c(self):
                    import cv2
                    image = np.full((5, 7), 3, np.uint8)
                    with self.assertRaises(cv2.error):
                        bad_resize(image)
                """
            )
        )
    elif question == "q120_open_video":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    try:
                        open_video(str(CASE / 'missing-b.avi'))
                    except RuntimeError as exc:
                        if type(exc) is not RuntimeError:
                            raise
                    else:
                        self.fail('RuntimeError not raised')

                def test_case_c(self):
                    capture = open_video(str(ROOT / 'fixtures' / 'clip.avi'))
                    self.assertTrue(capture.isOpened())
                    capture.release()
                """
            )
        )
    elif question == "q123_read_frame_with_retry":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    class FakeCap:
                        def read(self):
                            return True, np.full((4, 6, 3), 7, np.uint8)

                    actual = read_frame_with_retry(FakeCap(), 2)
                    np.testing.assert_array_equal(actual, np.full((4, 6, 3), 7, np.uint8))

                def test_case_c(self):
                    class FakeCap:
                        def read(self):
                            return False, None

                    with self.assertRaises(RuntimeError):
                        read_frame_with_retry(FakeCap(), 3)
                """
            )
        )
    elif question == "q124_load_or_raise":
        image = pattern(16, 16, (40, 50, 60))
        write_png(case_dir / "case_b.png", image)
        np.save(case_dir / "case_b_expected.npy", impl.load_or_raise(str(case_dir / "case_b.png")))
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    actual = load_or_raise(str(CASE / 'case_b.png'))
                    np.testing.assert_array_equal(actual, np.load(CASE / 'case_b_expected.npy'))

                def test_case_c(self):
                    with self.assertRaises(OpenCVError):
                        load_or_raise(str(CASE / 'missing-c.png'))
                """
            )
        )
    elif question == "q125_pipeline":
        image = pattern(28, 28, (70, 80, 90))
        write_png(case_dir / "case_c.png", image)
        np.save(case_dir / "case_c_expected.npy", impl.pipeline(str(case_dir / "case_c.png")))
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    self.assertIsNone(pipeline(str(CASE / 'missing-b.png')))

                def test_case_c(self):
                    actual = pipeline(str(CASE / 'case_c.png'))
                    np.testing.assert_array_equal(actual, np.load(CASE / 'case_c_expected.npy'))
                """
            )
        )
    elif question == "q126_log_error":
        methods.append(
            textwrap.dedent(
                """\
                def test_case_b(self):
                    folder = tempfile.mkdtemp()
                    path = str(Path(folder) / 'errors.log')
                    log_error(path, 'first')
                    log_error(path, 'second')
                    self.assertEqual(Path(path).read_text(encoding='utf-8'), 'first\\nsecond\\n')

                def test_case_c(self):
                    folder = tempfile.mkdtemp()
                    path = str(Path(folder) / 'errors.log')
                    log_error(path, '')
                    self.assertEqual(Path(path).read_text(encoding='utf-8'), '\\n')
                """
            )
        )
    else:
        return "unhandled"

    indented = "\n".join(
        textwrap.indent(part.strip() + "\n", "    ") if not part.startswith("    ") else part
        for part in methods
    )
    # methods were dedented to column 0. Indent the whole block by 4 spaces.
    block = "\n\n".join(textwrap.indent(part.strip("\n"), "    ") for part in methods)
    append_methods(path, block + "\n", imports)
    return "special"


def main() -> None:
    specials = {
        "q001_read_image",
        "q004_save_image",
        "q012_rotation_matrix",
        "q044_read_frames",
        "q045_video_properties",
        "q046_write_video",
        "q048_extract_every_nth",
        "q050_grayscale_video",
        "q081_camera_matrix",
        "q083_mean_reprojection_error",
        "q084_calibrate_views",
        "q085_stereo_rotation",
        "q111_image_table",
        "q117_labeled_table",
        "q118_load_image",
        "q119_bad_resize",
        "q120_open_video",
        "q123_read_frame_with_retry",
        "q124_load_or_raise",
        "q125_pipeline",
        "q126_log_error",
    }
    counts: dict[str, list[str]] = {}
    for path in sorted(TESTS.glob("test_q*.py")):
        question = path.stem.removeprefix("test_")
        text = path.read_text(encoding="utf-8")
        if "def test_case_b" in text:
            counts.setdefault("skipped", []).append(question)
            continue
        function_name = question.split("_", 1)[1]
        if question in specials:
            kind = extend_special(path, question)
        else:
            parsed = parse_method(text)
            if parsed is None:
                kind = "unparsed"
            else:
                kind = extend_simple(path, function_name, parsed)
        counts.setdefault(kind, []).append(question)
    for kind, names in sorted(counts.items()):
        print(f"{kind}: {len(names)}")
        if kind not in {"simple", "special", "skipped"}:
            print(" ", ", ".join(names))


if __name__ == "__main__":
    main()
