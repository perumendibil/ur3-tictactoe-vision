"""Day 2 tool: take the 15-20 chessboard photos for the lens calibration.

Live view with the detected inner corners drawn. Only frames where the whole
pattern is found can be saved. Photos are saved WITHOUT undistortion (raw),
because they are the input of calibrate_lens.py.

Keys:  space = save (only if the pattern is found)   q/Esc = quit
--auto N saves by itself when the board has moved enough, until N photos.

Examples (PowerShell, from calib\\):
    & $PY capture_chessboard.py --camera 0 --cols 9 --rows 6
    & $PY capture_chessboard.py --camera 1 --width 1280 --height 720 --auto 20

--cols/--rows count INNER corners (a 10x7-square board has 9x6 inner corners).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vision", "tictactoe"))
import camera  # noqa: E402

DEFAULT_DIR = os.path.join(HERE, "fotos", f"lente_{time.strftime('%Y-%m-%d')}")


def find_pattern(gray, cols, rows, fast=True):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    return found, corners


def main():
    parser = argparse.ArgumentParser(description="Capture chessboard photos for the lens calibration.")
    camera.add_camera_arguments(parser)
    parser.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
    parser.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
    parser.add_argument("--out", default=DEFAULT_DIR, help="folder for the photos (outside git)")
    parser.add_argument("--auto", type=int, default=0, help="save automatically until N photos")
    parser.add_argument("--min-move", type=float, default=0.08,
                        help="with --auto: minimum board move between photos (share of image width)")
    args = parser.parse_args()
    if args.calib:
        raise SystemExit("Do not use --calib here: the lens photos must be raw")

    cam = camera.camera_from_args(args)
    print(cam.info())
    os.makedirs(args.out, exist_ok=True)
    saved, last_centre, last_time = 0, None, 0.0
    title = "chessboard (space = save, q = quit)"
    while True:
        frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_pattern(gray, args.cols, args.rows)
        shown = frame.copy()
        cv2.drawChessboardCorners(shown, (args.cols, args.rows), corners, found)
        status = f"{'FOUND' if found else 'not found'}  saved {saved}"
        colour = (0, 200, 0) if found else (0, 0, 255)
        cv2.putText(shown, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        cv2.imshow(title, shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        want = key == ord(" ")
        if args.auto and found:
            centre = corners.reshape(-1, 2).mean(axis=0)
            moved = (last_centre is None or
                     np.linalg.norm(centre - last_centre) > args.min_move * frame.shape[1])
            want = want or (moved and time.time() - last_time > 1.0)
        if want and found:
            path = camera.save_frame(frame, args.out, "tablero")
            saved += 1
            last_centre = corners.reshape(-1, 2).mean(axis=0)
            last_time = time.time()
            print(f"{saved}: {path}")
            if args.auto and saved >= args.auto:
                break
        elif want:
            print("Pattern not found: photo not saved")
    cam.release()
    cv2.destroyAllWindows()
    print(f"{saved} photos in {args.out}")
    print(f"Next: & $PY calibrate_lens.py {args.out} --cols {args.cols} --rows {args.rows} --square <mm>")


if __name__ == "__main__":
    main()
