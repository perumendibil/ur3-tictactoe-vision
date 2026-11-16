"""Day 2 tool: lens calibration (K and dist) from the chessboard photos.

Calibrate (new K and dist):
    & $PY calibrate_lens.py fotos\\lente_2026-11-20 --cols 9 --rows 6 --square 25
    -> saves camera_<date>.npz here (K, dist, size, rms, square_mm, photos)

Check an existing calibration on new photos (no new K):
    & $PY calibrate_lens.py fotos\\check --cols 9 --rows 6 --square 25 --check camera_2026-11-20.npz

Quality (reprojection error, px): <= 0.5 very good, <= 1 acceptable, > 1 repeat.
--square is the measured side of one chessboard square, in mm.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def verdict(error_px):
    if error_px <= 0.5:
        return "very good"
    if error_px <= 1.0:
        return "acceptable"
    return "too high: remove blurry photos, add variety and repeat"


def board_points(cols, rows, square_mm):
    """3D points of the inner corners on the board plane (Z = 0), in mm."""
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return pts


def find_corners(paths, cols, rows):
    """Corners (sub-pixel) of every photo where the whole pattern is found."""
    found_paths, image_points, size = [], [], None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  skip {os.path.basename(path)}: cannot read")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        this_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = this_size
        elif this_size != size:
            print(f"  skip {os.path.basename(path)}: {this_size} px, others are {size} px")
            continue
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok:
            print(f"  skip {os.path.basename(path)}: pattern not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        found_paths.append(path)
        image_points.append(corners)
    return found_paths, image_points, size


def per_image_errors(obj, image_points, K, dist, rvecs, tvecs):
    errors = []
    for img_pts, rvec, tvec in zip(image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = img_pts.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt((diff ** 2).sum(axis=1).mean())))
    return errors


def calibrate(paths, cols, rows, square_mm):
    found_paths, image_points, size = find_corners(paths, cols, rows)
    if len(found_paths) < 3:
        raise SystemExit(f"Only {len(found_paths)} usable photos: at least 10 are needed (15-20 better)")
    obj = board_points(cols, rows, square_mm)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj] * len(image_points), image_points, size, None, None)
    errors = per_image_errors(obj, image_points, K, dist, rvecs, tvecs)
    return dict(rms=rms, K=K, dist=dist, size=size, paths=found_paths, errors=errors)


def check(paths, cols, rows, square_mm, K, dist):
    """Reprojection error of an existing K, dist on new photos (pose from solvePnP)."""
    found_paths, image_points, _ = find_corners(paths, cols, rows)
    obj = board_points(cols, rows, square_mm)
    rvecs, tvecs = [], []
    for img_pts in image_points:
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return found_paths, per_image_errors(obj, image_points, K, dist, rvecs, tvecs)


def list_photos(folder):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(PHOTO_EXTENSIONS))
    if not paths:
        raise SystemExit(f"No photos in {folder}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Lens calibration with a chessboard.")
    parser.add_argument("folder", help="folder with the chessboard photos")
    parser.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
    parser.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
    parser.add_argument("--square", type=float, required=True, help="side of one square, in mm")
    parser.add_argument("--out", help="output .npz (default calib/camera_<date>.npz)")
    parser.add_argument("--check", metavar="NPZ", help="only check this existing calibration")
    parser.add_argument("--undistorted", help="save the first photo undistorted to this file")
    args = parser.parse_args()
    paths = list_photos(args.folder)

    if args.check:
        data = np.load(args.check)
        K, dist = data["K"], data["dist"]
        found, errors = check(paths, args.cols, args.rows, args.square, K, dist)
        for p, e in zip(found, errors):
            print(f"  {os.path.basename(p)}: {e:.3f} px")
        if not errors:
            raise SystemExit("Pattern not found in any photo")
        mean = float(np.mean(errors))
        print(f"Mean reprojection error: {mean:.3f} px -> {verdict(mean)}")
        return

    result = calibrate(paths, args.cols, args.rows, args.square)
    K, dist = result["K"], result["dist"]
    print(f"Photos used: {len(result['paths'])} of {len(paths)}  ({result['size'][0]}x{result['size'][1]} px)")
    worst = sorted(zip(result["errors"], result["paths"]), reverse=True)[:3]
    for e, p in worst:
        print(f"  worst: {os.path.basename(p)}: {e:.3f} px")
    print(f"RMS reprojection error: {result['rms']:.3f} px -> {verdict(result['rms'])}")
    print("K (px):\n", np.array2string(K, precision=2, suppress_small=True))
    print("dist [k1 k2 p1 p2 k3]:", np.array2string(dist.ravel(), precision=4))

    out = args.out or os.path.join(HERE, f"camera_{time.strftime('%Y-%m-%d')}.npz")
    np.savez(out, K=K, dist=dist, size=np.array(result["size"]), rms=result["rms"],
             square_mm=args.square, pattern=np.array([args.cols, args.rows]),
             photos=len(result["paths"]), date=time.strftime("%Y-%m-%d"))
    print(f"Saved {out}")
    if args.undistorted:
        image = cv2.imread(result["paths"][0])
        cv2.imwrite(args.undistorted, cv2.hconcat([image, cv2.undistort(image, K, dist)]))
        print(f"Before / after saved to {args.undistorted}")


if __name__ == "__main__":
    main()
