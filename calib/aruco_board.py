"""Day 1-2 tool: find the ArUco markers of the board and turn them into
(a) the homography H from pixels to mm and (b) the 4 board corners in px for
vision/tictactoe/board_config.json.

Step 1, just look (which dictionary, which ids, how big in px):
    & $PY aruco_board.py --image foto.png --dict DICT_4X4_50 --debug aruco_debug.png
    & $PY aruco_board.py --camera 0 --try-all        # tries every 4x4..7x7 dictionary

Step 2, with a layout file (where each marker centre is, in mm; see
aruco_layout_example.json), compute H, the error per marker in mm and the
board corners:
    & $PY aruco_board.py --image foto.png --layout aruco_layout.json --save-h H.npy --write-config

Method (DECISIONES.md, 2026-10-22): sub-pixel corners, marker centre from the
crossing of its diagonals, markers matched by id, findHomography WITHOUT
RANSAC, error of every marker printed so a badly measured one shows up.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TICTACTOE = os.path.join(HERE, "..", "vision", "tictactoe")
sys.path.insert(0, TICTACTOE)
import camera  # noqa: E402
import config_tools  # noqa: E402

DEFAULT_BOARD_CONFIG = os.path.normpath(os.path.join(TICTACTOE, "board_config.json"))
TRY_DICTS = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_5X5_50", "DICT_5X5_100",
             "DICT_6X6_50", "DICT_6X6_250", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL"]


def make_detector(dict_name):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("This OpenCV has no aruco module: pip install opencv-contrib-python")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def diagonal_centre(c):
    """Crossing of the diagonals c0-c2 and c1-c3 (better than the mean with perspective)."""
    p0, p1, p2, p3 = [np.asarray(v, dtype=np.float64) for v in c]
    d1, d2 = p2 - p0, p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return c.mean(axis=0)
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    return p0 + t * d1


def detect(image, dict_name):
    """{id: (centre_px, corners_px)} for every marker found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = make_detector(dict_name).detectMarkers(gray)
    found = {}
    if ids is None:
        return found
    for c, i in zip(corners, ids.ravel()):
        c = c.reshape(4, 2)
        found[int(i)] = (diagonal_centre(c), c)
    return found


def side_px(c):
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def homography(found, markers_mm):
    """H (pixels -> mm) from the markers present in both, plus the error of each one in mm."""
    ids = sorted(i for i in found if str(i) in markers_mm)
    if len(ids) < 4:
        raise SystemExit(f"Only {len(ids)} markers of the layout were found ({ids}): 4 or more are needed")
    src = np.array([found[i][0] for i in ids], dtype=np.float64)
    dst = np.array([markers_mm[str(i)] for i in ids], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)  # 0 = plain least squares, no RANSAC
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = {i: float(np.linalg.norm(m - d)) for i, m, d in zip(ids, mapped, dst)}
    return H, errors


def board_corners_px(H, corners_mm):
    """The 4 board corners given in mm -> pixels (inverse homography)."""
    pts = np.array(corners_mm, dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
    return [[int(round(x)), int(round(y))] for x, y in px]


def draw(image, found, corners=None):
    shown = image.copy()
    for i, (centre, c) in found.items():
        cv2.polylines(shown, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.circle(shown, tuple(int(v) for v in centre), 4, (0, 0, 255), -1)
        cv2.putText(shown, str(i), (int(c[0][0]), int(c[0][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if corners:
        cv2.polylines(shown, [np.array(corners, np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
    return shown


def load_layout(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description="ArUco markers -> H (px to mm) and board corners.")
    camera.add_camera_arguments(parser)
    parser.add_argument("--image", help="photo to use instead of the camera")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary, e.g. DICT_4X4_50")
    parser.add_argument("--try-all", action="store_true", help="try the usual dictionaries and report")
    parser.add_argument("--layout", help="JSON with the marker centres in mm (see aruco_layout_example.json)")
    parser.add_argument("--save-h", help="save H (pixels -> mm) to this .npy file")
    parser.add_argument("--write-config", action="store_true",
                        help="write the board corners into board_config.json")
    parser.add_argument("--config", default=DEFAULT_BOARD_CONFIG, help="board_config.json to update")
    parser.add_argument("--debug", help="save a picture with the markers drawn")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Could not open {args.image}")
        if args.calib:
            K, dist, size = camera.load_lens_calibration(args.calib)
            image = camera.Undistorter(K, dist, size or (image.shape[1], image.shape[0]))(image)
    else:
        cam = camera.camera_from_args(args)
        image = cam.read()
        cam.release()

    if args.try_all:
        for name in TRY_DICTS:
            found = detect(image, name)
            if found:
                print(f"{name}: ids {sorted(found)}")
        print("Use the dictionary that finds the expected ids (and no strange extra ones).")
        return

    found = detect(image, args.dict)
    if not found:
        raise SystemExit(f"No {args.dict} markers found. Try --try-all, more light or a closer photo.")
    print(f"{len(found)} markers ({args.dict}):")
    for i in sorted(found):
        centre, c = found[i]
        print(f"  id {i}: centre ({centre[0]:.1f}, {centre[1]:.1f}) px, side {side_px(c):.1f} px")

    corners = None
    if args.layout:
        layout = load_layout(args.layout)
        H, errors = homography(found, layout["markers_mm"])
        for i, e in errors.items():
            flag = "  <- check this marker" if e > 2.0 else ""
            print(f"  error id {i}: {e:.2f} mm{flag}")
        print(f"Max error {max(errors.values()):.2f} mm (with exactly 4 markers it is always ~0)")
        if args.save_h:
            np.save(args.save_h, H)
            print(f"H saved to {args.save_h}")
        if layout.get("board_corners_mm"):
            corners = board_corners_px(H, layout["board_corners_mm"])
    elif args.write_config:
        raise SystemExit("--write-config needs --layout (with board_corners_mm)")

    if corners:
        print("Board corners (px):", json.dumps(corners))
        if args.write_config:
            config_tools.update_config_file(args.config, {"board_corners_px": corners})
            print(f"Written to {args.config} as board_corners_px")
    if args.debug:
        cv2.imwrite(args.debug, draw(image, found, corners))
        print(f"Debug picture saved to {args.debug}")


if __name__ == "__main__":
    main()
