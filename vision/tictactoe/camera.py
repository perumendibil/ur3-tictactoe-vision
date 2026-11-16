"""Open the camera (or a video / folder of photos) and give frames.

One small place for everything camera-related, so every tool opens the camera
the same way:

    from camera import Camera
    cam = Camera(0, width=1280, height=720, calib="../../calib/camera_2026-11-20.npz")
    frame = cam.read()          # BGR image, already undistorted if calib is given
    cam.release()

The source can be:
  - a number (0, 1, 2...)        -> a USB camera (on Windows, DirectShow is used)
  - a path to a video file       -> its frames one by one
  - a path to a folder of photos -> the .jpg/.png files in name order (loops)
  - a path to one photo          -> the same photo every time
"""

import glob
import os
import sys
import time

import cv2
import numpy as np

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_lens_calibration(path):
    """Read K (camera matrix) and dist (distortion) saved by calib/calibrate_lens.py.

    Returns (K, dist, size) where size is (width, height) in px or None.
    """
    data = np.load(path)
    size = tuple(int(v) for v in data["size"]) if "size" in data.files else None
    return data["K"], data["dist"], size


class Undistorter:
    """Remove lens distortion fast: the correction map is computed once (remap)."""

    def __init__(self, K, dist, size):
        self.K, self.dist, self.size = K, dist, tuple(size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, self.size, cv2.CV_16SC2)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(
                f"Frame is {w}x{h} px but the lens calibration was made at "
                f"{self.size[0]}x{self.size[1]} px: set the same resolution or recalibrate")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


def parse_source(text):
    """ "0" -> 0 (camera index); anything else stays a path."""
    return int(text) if str(text).isdigit() else text


class Camera:
    """A camera, a video, a folder of photos or one photo, with the same read()."""

    def __init__(self, source=0, width=None, height=None, calib=None,
                 autofocus=None, focus=None, exposure=None, auto_wb=None, wb_temperature=None):
        self.source = parse_source(source)
        self.cap = None
        self.photos = None
        self.index = 0

        if isinstance(self.source, str) and os.path.isdir(self.source):
            files = sorted(f for f in glob.glob(os.path.join(self.source, "*"))
                           if f.lower().endswith(PHOTO_EXTENSIONS))
            if not files:
                raise SystemExit(f"No photos in folder {self.source}")
            self.photos = files
        elif isinstance(self.source, str) and self.source.lower().endswith(PHOTO_EXTENSIONS):
            if not os.path.isfile(self.source):
                raise SystemExit(f"Photo not found: {self.source}")
            self.photos = [self.source]
        else:
            if isinstance(self.source, int) and sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise SystemExit(
                    f"Could not open camera/video {self.source!r}. Try another index "
                    "(0, 1, 2) and close other programs that use the camera.")
            props = [
                (cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                (cv2.CAP_PROP_AUTOFOCUS, autofocus), (cv2.CAP_PROP_FOCUS, focus),
                (cv2.CAP_PROP_EXPOSURE, exposure), (cv2.CAP_PROP_AUTO_WB, auto_wb),
                (cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature),
            ]
            for prop, value in props:
                if value is not None:
                    self.cap.set(prop, value)

        self.undistort = None
        if calib:
            K, dist, size = load_lens_calibration(calib)
            if size is None:
                first = self._raw_read()
                size = (first.shape[1], first.shape[0])
            self.undistort = Undistorter(K, dist, size)

    def _raw_read(self):
        if self.photos is not None:
            path = self.photos[self.index % len(self.photos)]
            self.index += 1
            frame = cv2.imread(path)
            if frame is None:
                raise SystemExit(f"Could not read {path}")
            return frame
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise SystemExit("The camera gave no frame (unplugged, or end of the video)")
        return frame

    @property
    def last_name(self):
        """File name of the last photo read (only for folders of photos)."""
        if self.photos is None or self.index == 0:
            return None
        return os.path.basename(self.photos[(self.index - 1) % len(self.photos)])

    def read(self):
        """Next frame (BGR), undistorted if a lens calibration was given."""
        frame = self._raw_read()
        return self.undistort(frame) if self.undistort else frame

    def info(self):
        """Short text: what is open, resolution in px and FPS reported by the driver."""
        if self.cap is None:
            return f"photos: {len(self.photos)} file(s)"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        focus = self.cap.get(cv2.CAP_PROP_FOCUS)
        return f"source {self.source!r}: {w}x{h} px, driver FPS {fps:.1f}, focus {focus:.0f}"

    def release(self):
        if self.cap is not None:
            self.cap.release()


def timestamp():
    """2026-11-20_10-31-05 (safe in Windows file names)."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def save_frame(frame, folder, prefix="lab"):
    """Save a frame as <folder>/<prefix>_<date_time>.png and return the path."""
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, f"{prefix}_{timestamp()}")
    path, n = base + ".png", 1
    while os.path.exists(path):  # two photos in the same second
        path, n = f"{base}_{n}.png", n + 1
    cv2.imwrite(path, frame)
    return path


def add_camera_arguments(parser):
    """The same camera options for every tool."""
    parser.add_argument("--camera", default="0",
                        help="camera index (0, 1, 2), video file, photo or folder of photos")
    parser.add_argument("--width", type=int, help="ask the camera for this width in px")
    parser.add_argument("--height", type=int, help="ask the camera for this height in px")
    parser.add_argument("--calib", help="lens calibration .npz (from calib/calibrate_lens.py)")
    parser.add_argument("--focus", type=float,
                        help="fixed focus value (turns autofocus off), if the camera allows it")


def camera_from_args(args):
    autofocus = 0 if args.focus is not None else None
    return Camera(args.camera, args.width, args.height, args.calib,
                  autofocus=autofocus, focus=args.focus)
