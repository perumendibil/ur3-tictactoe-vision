"""Read a tic-tac-toe board from a photo: blue cross = "X", red circle = "O".

Steps (like a signal chain, one stage after another):
  1. Straighten the board: if we know its 4 corners in the photo, warp it to
     a square seen from above (perspective correction, cv2.warpPerspective).
  2. Cut the square into 3 x 3 squares and keep only the middle of each one
     (cell_margin), so the grid lines and neighbour pieces do not count.
  3. In each square, count the share of red and blue pixels (HSV thresholds).
  4. Decide: below min_fraction -> empty; otherwise the stronger colour wins.

All thresholds live in board_config.json, so they can be tuned at ASSAR
without touching the code.

Usage from the command line:
    python board_reader.py photo.jpg
    python board_reader.py photo.jpg --debug debug.png
    python board_reader.py photo.jpg --corners "150,70 500,95 530,430 120,400"
    python board_reader.py photo.jpg --pick      # click the 4 corners with the mouse
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

import game

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "board_config.json")


@dataclass
class BoardReading:
    """What the reader saw.

    grid      : 3x3 list of rows with "X", "O" or "."
    red       : 3x3 share of red pixels in each square (0.0 to 1.0)
    blue      : 3x3 share of blue pixels in each square
    doubtful  : list of square numbers (1-9) where red and blue were close
    top_view  : the straightened board image (useful for debugging)
    """
    grid: list
    red: list
    blue: list
    doubtful: list = field(default_factory=list)
    top_view: np.ndarray = None

    @property
    def board(self):
        """The same reading as a flat list of 9 squares (see game.py)."""
        return game.from_grid(self.grid)


def load_config(path=DEFAULT_CONFIG):
    """Read the JSON settings. Keys starting with "_" are comments and are skipped."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def warp_board(image, corners, size_px=300):
    """Straighten the board to a size_px x size_px top view.

    corners: 4 (x, y) pixel points: top-left, top-right, bottom-right,
    bottom-left. Later this can come from the ArUco markers of the ASSAR
    board instead of being typed into the config.
    """
    src = np.array(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [size_px - 1, 0], [size_px - 1, size_px - 1], [0, size_px - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, H, (size_px, size_px))


def colour_mask(hsv, ranges):
    """White (255) where the pixel is inside any of the HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        low = np.array(r["low"], dtype=np.uint8)
        high = np.array(r["high"], dtype=np.uint8)
        mask |= cv2.inRange(hsv, low, high)
    return mask


def check_image_size(image, cfg):
    """Raise ValueError if the photo is not the size the board corners were taken at."""
    expected = cfg.get("image_size_px")
    if not (expected and cfg.get("board_corners_px")):
        return
    height, width = image.shape[:2]
    if [width, height] != list(expected):
        raise ValueError(
            f"The photo is {width}x{height} px but board_corners_px are for "
            f"{expected[0]}x{expected[1]} px. Use --width {expected[0]} --height {expected[1]}.")


def read_board(image, config=None):
    """Read the 3x3 board from a BGR photo (as given by cv2.imread or a camera).

    Returns a BoardReading. Raises ValueError if the image is empty or is not
    the size of image_size_px.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image (wrong path, or the camera gave no frame?)")
    cfg = config or load_config()
    check_image_size(image, cfg)
    size = int(cfg["warp_size_px"])

    if cfg.get("board_corners_px"):
        top = warp_board(image, cfg["board_corners_px"], size)
    else:
        top = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    blur = int(cfg.get("blur_px", 0))
    if blur >= 3:
        blur += (blur + 1) % 2  # the blur size must be odd
        top_blur = cv2.GaussianBlur(top, (blur, blur), 0)
    else:
        top_blur = top
    hsv = cv2.cvtColor(top_blur, cv2.COLOR_BGR2HSV)
    red_mask = colour_mask(hsv, cfg["red_ranges"])
    blue_mask = colour_mask(hsv, cfg["blue_ranges"])

    step = size / 3.0
    margin = int(step * float(cfg["cell_margin"]))
    min_fraction = float(cfg["min_fraction"])
    ambiguous_ratio = float(cfg["ambiguous_ratio"])
    # Defaults switch the checks off, so older configs behave as before.
    trace_fraction = float(cfg.get("trace_fraction", 1.0))
    piece_low, piece_high = cfg.get("piece_fraction_range", [0.0, 1.0])
    # The white squares and pieces are always there, so the brightest 10 % of the board
    # tells if there is enough light. Too dark: every square doubtful, not "empty".
    too_dark = np.percentile(hsv[..., 2], 90) < float(cfg.get("min_brightness", 0))

    grid, red, blue, doubtful = [], [], [], []
    for row in range(3):
        grid_row, red_row, blue_row = [], [], []
        for col in range(3):
            y0, y1 = int(row * step) + margin, int((row + 1) * step) - margin
            x0, x1 = int(col * step) + margin, int((col + 1) * step) - margin
            area = max(1, (y1 - y0) * (x1 - x0))
            r = cv2.countNonZero(red_mask[y0:y1, x0:x1]) / area
            b = cv2.countNonZero(blue_mask[y0:y1, x0:x1]) / area
            strong, weak = max(r, b), min(r, b)
            if strong < min_fraction:
                symbol = game.EMPTY
                unsure = strong >= trace_fraction  # part of a piece sitting off-centre
            else:
                symbol = game.O if r > b else game.X
                unsure = (weak >= min_fraction and weak > ambiguous_ratio * strong) \
                    or not piece_low <= strong <= piece_high  # off-centre piece, or a hand on it
            if unsure or too_dark:
                doubtful.append(row * 3 + col + 1)
            grid_row.append(symbol)
            red_row.append(round(r, 3))
            blue_row.append(round(b, 3))
        grid.append(grid_row)
        red.append(red_row)
        blue.append(blue_row)

    return BoardReading(grid=grid, red=red, blue=blue, doubtful=doubtful, top_view=top)


def draw_debug(reading, config=None):
    """Picture for checking the reader: top view + grid + what was read.

    Left: the straightened board with the area that is measured in each square
    and the result. Right: the red and blue masks (what the thresholds keep).
    """
    cfg = config or load_config()
    top = reading.top_view.copy()
    size = top.shape[0]
    step = size / 3.0
    margin = int(step * float(cfg["cell_margin"]))
    for row in range(3):
        for col in range(3):
            x0, y0 = int(col * step), int(row * step)
            cv2.rectangle(top, (x0 + margin, y0 + margin),
                          (int(x0 + step) - margin, int(y0 + step) - margin), (0, 255, 0), 1)
            symbol = reading.grid[row][col]
            cell = row * 3 + col + 1
            label = f"{cell}:{symbol}" + ("?" if cell in reading.doubtful else "")
            cv2.putText(top, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(top, label, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

    blur = int(cfg.get("blur_px", 0))
    src = reading.top_view
    if blur >= 3:
        blur += (blur + 1) % 2
        src = cv2.GaussianBlur(src, (blur, blur), 0)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    masks = np.zeros_like(top)
    masks[colour_mask(hsv, cfg["red_ranges"]) > 0] = (0, 0, 255)
    masks[colour_mask(hsv, cfg["blue_ranges"]) > 0] = (255, 0, 0)
    return cv2.hconcat([top, masks])


def print_reading(reading):
    """Print the board and the colour shares, to help tuning the thresholds."""
    print(game.pretty(reading.board))
    print()
    print("Share of red / blue pixels per square (min_fraction decides 'empty'):")
    for row in range(3):
        cells = []
        for col in range(3):
            cell = row * 3 + col + 1
            cells.append(f"{cell}: R {reading.red[row][col]:.2f} B {reading.blue[row][col]:.2f}")
        print("   ".join(cells))
    if reading.doubtful:
        print(f"Doubtful squares (red and blue close): {reading.doubtful}")


def parse_corners(text):
    """ "150,70 500,95 530,430 120,400" -> [[150, 70], [500, 95], [530, 430], [120, 400]]"""
    points = [[int(float(v)) for v in pair.split(",")] for pair in text.split()]
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise ValueError(f"Expected 4 points 'x,y', got: {text!r}")
    return points


def pick_corners(image):
    """Open a window and let the user click the 4 board corners.

    Order: top-left, top-right, bottom-right, bottom-left. Press any key when
    done (or Esc to cancel). Needs a screen (it will not work over SSH).
    """
    points = []
    shown = image.copy()
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    title = "Click the corners: " + ", ".join(names)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(shown, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(shown, names[len(points) - 1], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(title, shown)

    cv2.imshow(title, shown)
    cv2.setMouseCallback(title, on_click)
    while True:
        key = cv2.waitKey(50)
        if key == 27 or (key != -1 and len(points) == 4):
            break
    cv2.destroyAllWindows()
    if len(points) != 4:
        raise SystemExit("Cancelled: 4 corners are needed")
    return points


def main():
    parser = argparse.ArgumentParser(description="Read a tic-tac-toe board from a photo.")
    parser.add_argument("image", help="photo of the board (jpg, png...)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="settings file (JSON)")
    parser.add_argument("--debug", help="save a debug picture to this file")
    parser.add_argument("--corners", help='board corners in pixels, e.g. "150,70 500,95 530,430 120,400"')
    parser.add_argument("--pick", action="store_true", help="click the 4 board corners with the mouse")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not open {args.image}")
    cfg = load_config(args.config)
    if args.pick:
        cfg["board_corners_px"] = pick_corners(image)
        print("Corners (paste into board_config.json as board_corners_px):")
        print(json.dumps(cfg["board_corners_px"]))
    elif args.corners:
        cfg["board_corners_px"] = parse_corners(args.corners)
    reading = read_board(image, cfg)
    print_reading(reading)
    if args.debug:
        cv2.imwrite(args.debug, draw_debug(reading, cfg))
        print(f"Debug picture saved to {args.debug}")


if __name__ == "__main__":
    main()
