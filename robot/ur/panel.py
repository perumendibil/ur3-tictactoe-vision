"""Live panel for the pitch video: the wrist camera with the reading drawn on it, what
the director is doing, the board as read and a state disc (look and rules: pitch/VIDEO.md).

It wraps the eyes (look/forget) and is one more Announcer output (say), so the
director does not know it exists. A thread of its own redraws FPS times a second, so
the pulsing dot, the countdown ring and the seconds keep moving while the director is
busy with the arm. Run alone:  python panel.py            -> the numbers of partidas.csv
                               python panel.py sheet.jpg  -> every state in one image
"""

import csv
import math
import os
import re
import sys
import threading
import time

import cv2
import numpy as np

from eyes import Eyes  # noqa: F401  (also adds vision/tictactoe to the import path)
import board_reader    # noqa: E402  (vision/tictactoe)
import game            # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # plain OpenCV letters: uglier, but the robot still plays
    Image = None
    print("[PANEL] Pillow not installed, plain letters (pip install pillow)", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "partidas.csv")

W, H = 1600, 900                 # window; fits a 1920x1080 laptop screen with its title bar
FPS = 25
CAM = (40, 104, 1200, 675)       # the C920 frame at 1200/1920, x y w h
SIDE = (1272, 104, 288, 758)
BAR = (40, 803, 1200, 56)        # the four steps
MINI = (1294, 150, 244)          # board as read, x y side
ROWS_Y = 440                     # turn / last move
DISC = (1416, 705, 74)           # state disc, cx cy r
RADIUS = 28

# BGR. Blue = person, orange = robot, red = something wrong or moving.
INK, MUTE, CARD = (31, 29, 29), (139, 134, 134), (252, 250, 250)
BLUE, ORANGE, RED, AMBER = (220, 155, 58), (42, 138, 240), (48, 59, 255), (0, 190, 255)
GREY_RING, MINI_BG, MINI_EMPTY = (234, 229, 229), (220, 214, 214), (233, 229, 229)
STEPS = ("PERSON", "CAMERA", "DECISION", "ARM")
STEP_COLOUR = {"PERSON": BLUE, "CAMERA": ORANGE, "DECISION": ORANGE, "ARM": ORANGE}
# director state -> headline, colour, active step
STATES = {
    "YOUR_TURN": ("YOUR TURN", BLUE, "PERSON"),
    "GAME_OVER": ("GAME OVER", INK, "PERSON"),
    "COUNTDOWN": ("ROBOT PLAYS IN {n} s", ORANGE, "CAMERA"),
    "MOVEMENT": ("SOMETHING MOVES", RED, "CAMERA"),
    "UNSURE": ("CANNOT READ A SQUARE", AMBER, "CAMERA"),
    "ILLEGAL": ("BOARD CHANGED", RED, None),
    "BAD_BOARD": ("IMPOSSIBLE BOARD", RED, None),
    "ROBOT_MOVING": ("ROBOT MOVING", ORANGE, "ARM"),
    "PAUSED": ("PAUSED", RED, None),
}
PULSE_HZ = {"MOVEMENT": 2.5, "PAUSED": 0, "ILLEGAL": 0, "BAD_BOARD": 0}  # others 0.8


def stats(path=LOG):
    """Numbers from partidas.csv: games and results, O placed, s per move, last move."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        rows = []
    moves = [r for r in rows if r["event"] == "move"]
    results = [r["result"] for r in rows if r["event"] == "over"]
    placed = [float(r["seconds"]) for r in moves if r["result"] == "ok"]
    last = moves[-1] if moves else None
    return {"games": len(results), "won": results.count("O wins"),
            "draw": results.count("draw"), "lost": results.count("X wins"),
            "placed": len(placed), "moves": len(moves),
            "s_per_move": round(sum(placed) / len(placed)) if placed else None,
            "last_cell": int(last["cell"]) if last else None,
            "last_s": float(last["seconds"]) if last else None,
            "last_ok": last["result"] == "ok" if last else None}


def stats_line(s):
    per_move = f"{s['s_per_move']} s per move" if s["s_per_move"] is not None else "no moves yet"
    return (f"{s['games']} games: robot won {s['won']}, draw {s['draw']}, lost {s['lost']} | "
            f"O placed {s['placed']}/{s['moves']} | {per_move}")


# ---- drawing helpers ------------------------------------------------------

def rr_pts(x, y, w, h, r):
    """Rounded rectangle as a polygon (8 points per corner), for fillPoly/polylines."""
    r = max(1, min(r, w // 2, h // 2))
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                       (x + r, y + h - r, 90), (x + r, y + r, 180)):
        for k in range(9):
            a = math.radians(a0 + 90 * k / 8)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.int32)


def rrect(img, x, y, w, h, r, colour, thickness=-1):
    pts = rr_pts(x, y, w, h, r)
    if thickness < 0:
        cv2.fillPoly(img, [pts], colour, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, colour, thickness, cv2.LINE_AA)


def tint(img, pts, colour, alpha):
    """Translucent filled polygon; only the polygon's box is blended (fast)."""
    x0, y0 = np.clip(pts.min(axis=0), 0, (img.shape[1], img.shape[0]))
    x1, y1 = np.clip(pts.max(axis=0) + 1, 0, (img.shape[1], img.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    over = sub.copy()
    cv2.fillPoly(over, [pts - (x0, y0)], colour, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, sub, 1 - alpha, 0, sub)


_FONTS, _TEXT = {}, {}
FONT_FILES = {"r": ("segoeui.ttf", "DejaVuSans.ttf"), "s": ("seguisb.ttf", "DejaVuSans-Bold.ttf"),
              "b": ("segoeuib.ttf", "DejaVuSans-Bold.ttf")}


def _font(size, weight):
    key = (size, weight)
    if key not in _FONTS:
        for name in FONT_FILES[weight]:
            try:
                _FONTS[key] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def _patch(s, size, weight, colour):
    """The text rendered once: (BGR patch, alpha 0-1). Cached by its arguments."""
    key = (s, size, weight, colour)
    if key not in _TEXT:
        if len(_TEXT) > 400:  # the countdown and the seconds make new strings all the time
            _TEXT.clear()
        f = _font(size, weight)
        left, top, right, bottom = f.getbbox(s)
        im = Image.new("RGBA", (max(1, right + 2), max(1, bottom + 2)))
        ImageDraw.Draw(im).text((0, 0), s, font=f, fill=(colour[2], colour[1], colour[0], 255))
        rgba = np.asarray(im)
        _TEXT[key] = (np.ascontiguousarray(rgba[:, :, 2::-1]), rgba[:, :, 3:4].astype(np.float32) / 255)
    return _TEXT[key]


def measure(s, size, weight="r"):
    if Image is None:
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, size / 28, 1)[0][0]
    return _patch(s, size, weight, INK)[0].shape[1]


def text(img, s, xy, size, colour=INK, weight="r", anchor="l"):
    """Draw s with the top of its box at y and its left/centre/right at x."""
    x, y = int(xy[0]), int(xy[1])
    if Image is None:
        scale = size / 28
        w = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0][0]
        x -= {"l": 0, "c": w // 2, "r": w}[anchor]
        cv2.putText(img, s, (x, y + size), cv2.FONT_HERSHEY_DUPLEX, scale, colour, 1, cv2.LINE_AA)
        return
    bgr, alpha = _patch(s, size, weight, colour)
    h, w = alpha.shape[:2]
    x -= {"l": 0, "c": w // 2, "r": w}[anchor]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    sub[:] = (bgr[y0 - y:y1 - y, x0 - x:x1 - x] * a + sub * (1 - a)).astype(np.uint8)


def shadow(base, x, y, w, h, r, strength=0.22, blur=61):
    """Soft drop shadow under a card, baked into the static background."""
    mask = np.zeros(base.shape[:2], np.float32)
    cv2.fillPoly(mask, [rr_pts(x, y + 14, w, h, r)], 1.0, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)[:, :, None]
    base[:] = (base * (1 - mask * strength)).astype(np.uint8)


# ---- the panel ------------------------------------------------------------

class Panel:
    def __init__(self, eyes, log_path=LOG, cfg=None, title="Tic-tac-toe robot", window=True):
        self.eyes, self.log_path, self.title, self.window = eyes, log_path, title, window
        self.view, self.numbers = (None, None), stats(log_path)  # (last sight, its photo resized)
        self.state, self.prev, self.sentence, self.t_state = "", "", "", time.time()
        self.cell = None          # square the robot plays or played last, 1-9
        self.hint = None          # square the director complains about (UNSURE, PAUSED)
        self.t_count, self.count_n, self.countdown_s = None, 0, 5
        self.turn = ("", MUTE)
        self.thread = None
        cfg = cfg or board_reader.load_config()
        corners = cfg.get("board_corners_px")
        self.cells_px = self._cells_px(corners, cfg["warp_size_px"]) if corners else None
        self.base = self._base()
        mask = np.zeros((CAM[3], CAM[2]), np.float32)  # rounded corners of the camera card
        cv2.fillPoly(mask, [rr_pts(0, 0, CAM[2], CAM[3], RADIUS)], 1.0, cv2.LINE_AA)
        self.corner_mask = mask[:, :, None]

    @staticmethod
    def _cells_px(corners, size):
        """The 9 squares as 4-point polygons in the photo, from the board corners."""
        src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        back = cv2.getPerspectiveTransform(src, np.float32(corners))
        step = size / 3.0
        cells = []
        for row in range(3):
            for col in range(3):
                x0, y0 = col * step, row * step
                square = np.float32([[[x0, y0], [x0 + step, y0], [x0 + step, y0 + step], [x0, y0 + step]]])
                cells.append(cv2.perspectiveTransform(square, back)[0] * (CAM[2] / 1920.0))
        return cells

    # ---- what the director calls ------------------------------------------

    def look(self):
        sight = self.eyes.look()
        try:  # one assignment, so the drawing thread never pairs a board with the photo before
            self.view = (sight, cv2.resize(sight.frame, CAM[2:], interpolation=cv2.INTER_AREA))
        except Exception as err:  # a broken panel must never stop the robot
            print(f"[PANEL] {err}", flush=True)
        self._start()
        return sight

    def forget(self):
        self.eyes.forget()

    def say(self, state, sentence):
        now = time.time()
        if state != self.state:
            self.prev, self.state, self.t_state = self.state, state, now
        self.sentence = sentence
        digit = re.search(r"\d+", sentence)
        if state == "COUNTDOWN":
            n = int(digit.group()) if digit else self.count_n
            if self.t_count is None:
                self.countdown_s = n  # shortcut: the first number said is the whole wait
            left = self.countdown_s - (now - self.t_count) if self.t_count else 0
            if math.ceil(left) != n:  # new or restarted countdown: follow the director's number
                self.t_count = now - (self.countdown_s - n)
            self.count_n = n
        else:
            self.t_count = None
        if state == "ROBOT_MOVING":
            found = re.search(r"square (\d)", sentence)  # shortcut: the square comes from the sentence
            self.cell = int(found.group(1)) if found else None
        if state in ("UNSURE", "PAUSED"):
            self.hint = int(digit.group()) if digit else None
        if state in ("YOUR_TURN", "GAME_OVER"):
            self.numbers = stats(self.log_path)  # a row may have been added
        if state == "YOUR_TURN":
            self.turn = ("Person · X", BLUE)
        elif state in ("COUNTDOWN", "ROBOT_MOVING"):
            self.turn = ("Robot · O", ORANGE)
        elif state == "GAME_OVER":
            result = re.match(r"(O wins|X wins|draw)", sentence)
            self.turn = (result.group(1) if result else "Game over",
                         ORANGE if sentence.startswith("O") else BLUE if sentence.startswith("X") else INK)
        self._start()

    # ---- the drawing thread -----------------------------------------------

    def _start(self):
        if self.window and self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        try:  # Windows at 125 % scale would blow the window up to 2000x1125 px and cut it
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (ImportError, AttributeError, OSError):
            pass
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.title, 0, 0)  # whole panel on screen for OBS to capture it
        while True:
            t = time.time()
            try:
                cv2.imshow(self.title, self.render(t))
                cv2.waitKey(1)
            except Exception as err:  # a broken window must never stop the robot
                print(f"[PANEL] {err}", flush=True)
                time.sleep(1)
            time.sleep(max(0.0, 1.0 / FPS - (time.time() - t)))

    # ---- rendering --------------------------------------------------------

    def _base(self):
        """Everything that never changes, painted once: background, shadows, cards, labels."""
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - 0.3 * W) / 1200) ** 2 + ((yy - 0.2 * H) / 700) ** 2).clip(0, 1)[:, :, None]
        base = (np.array([255, 255, 255], np.float32) * (1 - d) + np.array([231, 226, 226], np.float32) * d).astype(np.uint8)
        for x, y, w, h in (CAM, SIDE, BAR):
            shadow(base, x, y, w, h, RADIUS)
        rrect(base, *SIDE, RADIUS, CARD)
        rrect(base, *BAR, BAR[3] // 2, (250, 248, 248))
        mx, my, side = MINI
        text(base, "BOARD AS READ", (mx, my - 22), 12, MUTE, "s")
        rrect(base, mx, my, side, side, 20, MINI_BG)
        for i in range(9):
            x, y, s = self._mini_cell(i)
            rrect(base, x, y, s, s, 12, MINI_EMPTY)
        text(base, "TURN", (mx, ROWS_Y), 13, MUTE, "s")
        cv2.line(base, (mx, ROWS_Y + 36), (mx + side, ROWS_Y + 36), (236, 236, 236), 1)
        text(base, "LAST ROBOT MOVE", (mx, ROWS_Y + 48), 13, MUTE, "s")
        text(base, "UR3 · Logitech C920 · Python + OpenCV", (mx, SIDE[1] + SIDE[3] - 48), 13, INK, "s")
        text(base, "ASSAR, Högskolan i Skövde", (mx, SIDE[1] + SIDE[3] - 30), 12, MUTE)
        return base

    @staticmethod
    def _mini_cell(i):
        mx, my, side = MINI
        s = (side - 2 * 9 - 2 * 7) // 3
        row, col = divmod(i, 3)
        return mx + 9 + col * (s + 7), my + 9 + row * (s + 7), s

    def render(self, now=None):
        """One frame of the panel, from the last photo and the last thing the director said."""
        now = time.time() if now is None else now
        img = self.base.copy()
        state, sentence, (sight, photo) = self.state, self.sentence, self.view
        headline, colour, active = STATES.get(state, (state, MUTE, None))
        moving = state == "MOVEMENT"
        self._camera(img, state, sight, photo, moving)
        self._steps(img, state, active)
        self._side(img, state, sight, moving, now)
        self._banner(img, state, headline.format(n=self.count_n), colour, sentence, now)
        return img

    def _camera(self, img, state, sight, photo, moving):
        x, y, w, h = CAM
        canvas = photo.copy() if photo is not None else np.full((h, w, 3), 60, np.uint8)
        paused = state in ("ROBOT_MOVING", "PAUSED")
        if paused:
            canvas = cv2.convertScaleAbs(canvas, alpha=0.45)
        if self.cells_px is not None and sight is not None:
            self._cells(canvas, sight, moving, self.cell if state == "ROBOT_MOVING" else None)
        self._pill(canvas, 22, 20, "LIVE · WRIST CAMERA", dot=RED)
        if sight is not None:
            label = f"{min(sight.motion_percent, 99.9):.1f} % moving"
            pw = measure(label, 15, "s") + 32 + 132
            self._pill(canvas, w - 22 - pw, 20, label, fill=(48, 59, 255) if moving else None,
                       meter=(sight.motion_percent, self.eyes.motion_percent))
        if paused:
            label = "CAMERA PAUSED WHILE THE ARM MOVES" if state == "ROBOT_MOVING" \
                else "CAMERA PAUSED: WAITING FOR A PERSON"
            self._pill(canvas, (w - measure(label, 15, "s") - 32) // 2, h // 2 - 18, label)
        # rounded corners: paste whole, then blend the four corners with the mask
        img[y:y + h, x:x + w] = canvas
        for cx, cy in ((x, y), (x + w - RADIUS, y), (x, y + h - RADIUS), (x + w - RADIUS, y + h - RADIUS)):
            sub = img[cy:cy + RADIUS, cx:cx + RADIUS]
            m = self.corner_mask[cy - y:cy - y + RADIUS, cx - x:cx - x + RADIUS]
            sub[:] = (sub * m + self.base[cy:cy + RADIUS, cx:cx + RADIUS] * (1 - m)).astype(np.uint8)
        if moving:
            rrect(img, x + 2, y + 2, w - 4, h - 4, RADIUS - 2, RED, 5)
        else:
            rrect(img, x, y, w, h, RADIUS, (255, 255, 255), 1)  # bevel

    def _cells(self, canvas, sight, moving, target):
        """Each square: rounded frame and light tint in the colour the reader saw, its number."""
        for i, poly in enumerate(self.cells_px):
            # shortcut: the box around the (slightly skewed) square, so corners can be round;
            # upgrade: draw the real polygon if the camera ever looks at the board at an angle.
            poly = (poly - poly.mean(axis=0)) * 0.94 + poly.mean(axis=0)
            x0, y0 = poly.min(axis=0).astype(int)
            x1, y1 = poly.max(axis=0).astype(int)
            symbol = sight.board[i]
            colour, alpha, thick = (ORANGE, 0.14, 3) if symbol == game.O else (BLUE, 0.14, 3) if symbol == game.X \
                else ((200, 200, 200), 0.0, 1)
            if i + 1 in sight.doubtful:
                colour, alpha, thick = AMBER, 0.25, 3
            if i + 1 == target:
                colour, alpha, thick = RED, 0.20, 6
            if moving:
                colour = RED
            pts = rr_pts(x0, y0, x1 - x0, y1 - y0, 16)
            if alpha:
                tint(canvas, pts, colour, alpha)
            cv2.polylines(canvas, [pts], True, (30, 30, 30), thick + 2, cv2.LINE_AA)  # dark halo
            cv2.polylines(canvas, [pts], True, colour, thick, cv2.LINE_AA)
            text(canvas, str(i + 1), (x0 + 9, y0 + 5), 14, (255, 255, 255), "s")

    @staticmethod
    def _pill(canvas, x, y, label, dot=None, fill=None, meter=None):
        """Translucent dark pill over the photo, with an optional dot or motion meter."""
        tw = measure(label, 15, "s")
        w = tw + 32 + (19 if dot else 0) + (132 if meter else 0)
        tint(canvas, rr_pts(x, y, w, 36, 18), fill or (22, 20, 20), 0.85 if fill else 0.62)
        tx = x + 16
        if dot:
            cv2.circle(canvas, (tx + 4, y + 18), 4, dot, -1, cv2.LINE_AA)
            tx += 19
        text(canvas, label, (tx, y + 7), 15, (255, 255, 255), "s")
        if meter:
            value, limit = meter
            mx, my, mw = tx + tw + 12, y + 15, 120
            rrect(canvas, mx, my, mw, 6, 3, (120, 120, 120))
            rrect(canvas, mx, my, max(6, int(mw * min(value, 20) / 20)), 6, 3, (255, 255, 255))
            lx = mx + int(mw * limit / 20)
            cv2.line(canvas, (lx, my - 3), (lx, my + 9), (255, 255, 255), 2)

    def _steps(self, img, state, active):
        x, y, w, h = BAR
        sw = (w - 10) // 4
        done = STEPS.index(active) if active else -1
        if state == "ROBOT_MOVING":
            done = 3  # decision taken too
        for k, step in enumerate(STEPS):
            sx = x + 5 + k * sw
            if step == active:
                rrect(img, sx, y + 5, sw, h - 10, (h - 10) // 2, STEP_COLOUR[step])
                col = (255, 255, 255)
            else:
                col = INK if k < done else (166, 161, 161)
            text(img, step, (sx + sw // 2, y + 18), 15, col, "s", "c")

    def _side(self, img, state, sight, moving, now):
        mx, my, side = MINI
        if moving:
            rrect(img, mx - 3, my - 3, side + 6, side + 6, 23, RED, 4)
        if sight is not None:
            for i, symbol in enumerate(sight.board):
                x, y, s = self._mini_cell(i)
                if symbol == game.EMPTY:
                    continue
                rrect(img, x, y, s, s, 12, (255, 255, 255))
                cx, cy, r = x + s // 2, y + s // 2, int(s * 0.29)
                if symbol == game.O:
                    cv2.circle(img, (cx, cy), r, ORANGE, 8, cv2.LINE_AA)
                else:
                    cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), BLUE, 8, cv2.LINE_AA)
                    cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), BLUE, 8, cv2.LINE_AA)
            if state == "ROBOT_MOVING" and self.cell:
                x, y, s = self._mini_cell(self.cell - 1)
                rrect(img, x, y, s, s, 12, RED, 3)
        label, colour = self.turn
        text(img, label, (mx + side, ROWS_Y - 3), 17, RED if moving else colour, "s", "r")
        n = self.numbers
        if n["last_cell"]:
            value = f"O on {n['last_cell']}"
            sec = f" · {n['last_s']:.0f} s" if n["last_ok"] else " · not seen"
            text(img, sec, (mx + side, ROWS_Y + 45), 17, MUTE if n["last_ok"] else RED, "r", "r")
            text(img, value, (mx + side - measure(sec, 17), ROWS_Y + 45), 17, INK, "s", "r")
        else:
            text(img, "–", (mx + side, ROWS_Y + 45), 17, MUTE, "s", "r")
        self._disc(img, state, now)

    def _ring(self, img, frac, colour):
        cx, cy, r = DISC
        cv2.circle(img, (cx, cy), r, GREY_RING, 12, cv2.LINE_AA)
        if frac > 0:
            cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360 * min(frac, 1), colour, 12, cv2.LINE_AA)
            for a in (-90, -90 + 360 * min(frac, 1)):  # round caps
                cv2.circle(img, (int(cx + r * math.cos(math.radians(a))),
                                 int(cy + r * math.sin(math.radians(a)))), 6, colour, -1, cv2.LINE_AA)

    def _glyph(self, img, glyph, colour):
        cx, cy, r = DISC
        if glyph == "X":
            d, cy = int(r * 0.42), cy - 16
            cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), colour, 14, cv2.LINE_AA)
            cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), colour, 14, cv2.LINE_AA)
        else:
            text(img, glyph, (cx, cy - 104), 110, colour, "b", "c")

    def _disc(self, img, state, now):
        """The same circle, a different content per state (design of 2026-11-12)."""
        cx, cy, r = DISC
        caption = None
        if state == "YOUR_TURN":
            self._glyph(img, "X", BLUE)
            caption = "PLACE AN X"
        elif state == "COUNTDOWN" and self.t_count is not None:
            left = max(0.0, self.countdown_s - (now - self.t_count))
            self._ring(img, 1 - left / self.countdown_s if self.countdown_s else 1, ORANGE)
            text(img, f"{left:.1f}", (cx - 8, cy - 26), 40, INK, "b", "c")
            text(img, f"/ {self.countdown_s} s", (cx + 28, cy - 6), 18, MUTE, "s")
            caption = "ROBOT PLAYS IN"
        elif state == "MOVEMENT" and self.prev == "COUNTDOWN":
            self._ring(img, 0.02, RED)
            text(img, "0.0", (cx - 8, cy - 26), 40, INK, "b", "c")
            text(img, f"/ {self.countdown_s} s", (cx + 28, cy - 6), 18, MUTE, "s")
            caption = "RESTARTED"
        elif state == "MOVEMENT":
            self._glyph(img, "X", RED)
            caption = "HOLD STILL"
        elif state == "ROBOT_MOVING":
            text(img, str(self.cell or "?"), (cx, cy - 56), 72, ORANGE, "b", "c")
            caption = f"SQUARE · {now - self.t_state:.0f} s"
        elif state == "UNSURE":
            self._glyph(img, "?", AMBER)
            caption = f"SQUARE {self.hint}" if self.hint else "A SQUARE"
        elif state == "PAUSED":
            self._glyph(img, "!", RED)
            caption = f"SQUARE {self.hint}" if self.hint else "CHECK THE BOARD"
        elif state in ("ILLEGAL", "BAD_BOARD"):
            self._glyph(img, "!", RED)
            caption = "PUT THE BOARD BACK"
        elif state == "GAME_OVER":
            label, colour = self.turn
            text(img, label.upper(), (cx, cy - 22), 30, colour, "b", "c")
            caption = "CLEAR THE BOARD"
        if caption:
            text(img, caption, (cx, cy + 22), 11, MUTE, "s", "c")

    def _banner(self, img, state, headline, colour, sentence, now):
        if not state:
            return
        tw = max(measure(headline, 22, "b"), measure(sentence, 15))
        w, h = tw + 24 + 16 + 26, 72
        x, y = W // 2 - w // 2, 22
        rrect(img, x, y, w, h, h // 2, (250, 248, 248))
        rrect(img, x, y, w, h, h // 2, (236, 236, 236), 1)
        hz = PULSE_HZ.get(state, 0.8)
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * hz * now) if hz else 0.0
        dx, dy = x + 24 + 4, y + h // 2
        tint(img, rr_pts(dx - 20, dy - 20, 40, 40, 20), colour, 0.12 + 0.25 * pulse)  # glow
        cv2.circle(img, (dx, dy), 12, colour, -1, cv2.LINE_AA)
        cv2.circle(img, (dx, dy), int(17 + 4 * pulse), colour, 2, cv2.LINE_AA)
        text(img, headline, (x + 24 + 16 + 16, y + 12), 22, colour, "b")
        text(img, sentence, (x + 24 + 16 + 16, y + 42), 15, (84, 81, 81))


# ---- contact sheet: every state in one image ------------------------------

SHEET = (("YOUR_TURN", "Your turn (X)"),
         ("MOVEMENT", "Something is moving: waiting for a still board"),
         ("COUNTDOWN", "Robot plays in 5 s: hands away"),
         ("ROBOT_MOVING", "Robot plays square 6: keep away"),
         ("UNSURE", "I cannot read square(s) [5]: straighten the piece or take the hand away"),
         ("PAUSED", "I cannot see my O on square 6. Put it there by hand and check the O store"),
         ("BAD_BOARD", "Impossible board: 3 O and 1 X"),
         ("GAME_OVER", "O wins. Clear the board and refill the O store to play again"))


class StillEyes:
    """Eyes that always see the same thing: for the contact sheet and the tests."""

    def __init__(self, sight, motion_percent=5):
        self.sight, self.motion_percent = sight, motion_percent

    def look(self):
        return self.sight

    def forget(self):
        pass


def sheet(path, photo=None, log_path=LOG):
    """Render the 8 states with a photo (or grey) and tile them 4x2 at half size."""
    from eyes import Sight
    frame = cv2.imread(photo) if photo else np.full((1080, 1920, 3), 90, np.uint8)
    sight = Sight(game.from_text("OOOOO...."), [], False, 0.4, frame)
    p = Panel(StillEyes(sight), log_path, window=False)
    p.look()
    tiles = []
    for state, sentence in SHEET:
        if state == "MOVEMENT":
            p.say("COUNTDOWN", "Robot plays in 5 s: hands away")  # so the ring shows "restarted"
            sight.motion_percent, sight.moving = 18.2, True
        else:
            sight.motion_percent, sight.moving = 0.4, False
        if state == "UNSURE":
            sight.doubtful = [5]
        p.say(state, sentence)
        tiles.append(cv2.resize(p.render(p.t_state + 1.8), (W // 2, H // 2), interpolation=cv2.INTER_AREA))
        sight.doubtful = []
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, 8, 4)]
    cv2.imwrite(path, np.vstack(rows))
    return path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sheet(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        print(stats_line(stats()))
