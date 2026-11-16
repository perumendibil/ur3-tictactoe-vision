"""Draw fake photos of a tic-tac-toe board, to test the reader without a camera.

Blue crosses ("X") and red circles ("O") on a light board, optionally seen at
an angle (perspective), with uneven light and noise. The fake photo also
returns the pixel corners of the board, like ArUco markers would.

Usage:
    python synthetic.py "XO./.X./..O" fake.png
    python synthetic.py "XO./.X./..O" fake.png --tilt --noise 12
"""

import argparse

import cv2
import numpy as np

import game

# Colours in BGR (OpenCV order: blue, green, red).
BOARD_BGR = (215, 225, 230)
LINE_BGR = (40, 40, 40)
RED_BGR = (40, 40, 200)
BLUE_BGR = (190, 90, 30)

# Synthetic photos are not 1920x1080 px and their pieces are thick (about 40 % of a
# square, real ones about 10 %), so tests switch off those two checks of
# board_config.json: dict(cfg, **CONFIG).
CONFIG = {"piece_fraction_range": [0.0, 1.0], "image_size_px": None}


def draw_top_view(board, size_px=300, line_px=4, piece_px=10):
    """Board seen straight from above, size_px x size_px."""
    img = np.full((size_px, size_px, 3), BOARD_BGR, dtype=np.uint8)
    step = size_px / 3.0
    for k in (1, 2):
        p = int(k * step)
        cv2.line(img, (p, 0), (p, size_px), LINE_BGR, line_px)
        cv2.line(img, (0, p), (size_px, p), LINE_BGR, line_px)
    for i, symbol in enumerate(board):
        row, col = divmod(i, 3)
        cx, cy = int((col + 0.5) * step), int((row + 0.5) * step)
        r = int(step * 0.3)
        if symbol == game.O:
            cv2.circle(img, (cx, cy), r, RED_BGR, piece_px, cv2.LINE_AA)
        elif symbol == game.X:
            cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), BLUE_BGR, piece_px, cv2.LINE_AA)
            cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), BLUE_BGR, piece_px, cv2.LINE_AA)
    return img


def make_photo(board, tilt=False, noise=0.0, light_gradient=0.0, seed=0,
               photo_size=(640, 480)):
    """Fake camera photo of `board`.

    tilt           : put the board at an angle inside a bigger, darker table.
    noise          : standard deviation of random pixel noise (0-255 scale).
    light_gradient : 0.3 = one side of the photo 30 % darker than the other.

    Returns (photo, corners): corners are the 4 board corners in pixels
    (top-left, top-right, bottom-right, bottom-left).
    """
    rng = np.random.default_rng(seed)
    top = draw_top_view(board)
    s = top.shape[0]
    w, h = photo_size
    if tilt:
        corners = np.array([[150, 70], [500, 95], [530, 430], [120, 400]], dtype=np.float32)
    else:
        corners = np.array([[170, 90], [470, 90], [470, 390], [170, 390]], dtype=np.float32)
    src = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, corners)
    table = np.full((h, w, 3), (90, 110, 120), dtype=np.uint8)
    warped = cv2.warpPerspective(top, H, (w, h))
    inside = cv2.warpPerspective(np.full((s, s), 255, np.uint8), H, (w, h)) > 0
    photo = table.copy()
    photo[inside] = warped[inside]

    photo = photo.astype(np.float32)
    if light_gradient:
        ramp = np.linspace(1.0, 1.0 - light_gradient, w, dtype=np.float32)
        photo *= ramp[None, :, None]
    if noise:
        photo += rng.normal(0, noise, photo.shape).astype(np.float32)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    return photo, corners.tolist()


def main():
    parser = argparse.ArgumentParser(description="Draw a fake photo of a board.")
    parser.add_argument("board", help='board as text, e.g. "XO./.X./..O"')
    parser.add_argument("output", help="file to save (png or jpg)")
    parser.add_argument("--tilt", action="store_true", help="see the board at an angle")
    parser.add_argument("--noise", type=float, default=0.0, help="pixel noise, e.g. 10")
    parser.add_argument("--light", type=float, default=0.0, help="light gradient, e.g. 0.3")
    args = parser.parse_args()

    photo, corners = make_photo(game.from_text(args.board), args.tilt, args.noise, args.light)
    cv2.imwrite(args.output, photo)
    print(f"Saved {args.output}")
    print(f"Board corners in pixels (for board_corners_px): {[[int(x), int(y)] for x, y in corners]}")


if __name__ == "__main__":
    main()
