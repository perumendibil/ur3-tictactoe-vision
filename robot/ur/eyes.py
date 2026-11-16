"""The eyes of the director: one camera frame -> the board, and is anything moving?

It joins the camera and the board reader of vision/tictactoe and adds the one
thing they do not have: comparing each frame with the one before, so a hand over
the board (or the arm still swinging) is seen before the board is trusted.

Limit to remember: the camera is on the wrist. While the arm moves it does not
see the board, so this only protects BEFORE a move starts.
"""

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "vision", "tictactoe"))
import board_reader  # noqa: E402

# A pixel "changed" if its grey level moved more than this (0-255) between two frames.
# Lab photos 2026-11-10: the auto-exposure shifts the whole image by up to ~17.
PIXEL_CHANGE = 25
# Share of changed pixels (%) that counts as movement. Measured on lab photos: one
# piece pushed aside = 1.5 %, a hand = 16 % or more. Live C920, 2026-11-11: still board
# 0 % with lone spikes up to 3 % (they reset the countdown); a hand, median 18 %.
MOTION_PERCENT = 5
STALE_FRAMES = 5  # frames waiting in the camera queue after the arm was moving


@dataclass
class Sight:
    board: list            # 9 squares, "X", "O" or "." (see vision/tictactoe/game.py)
    doubtful: list         # squares 1-9 the reader is not sure about
    moving: bool
    motion_percent: float  # share of pixels that changed since the frame before
    frame: np.ndarray

    @property
    def clear(self):
        """True if this frame can be trusted: nothing moving, no doubtful square."""
        return not self.moving and not self.doubtful


def _small(frame):
    grey = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (5, 5), 0)  # the blur hides sensor noise


class Eyes:
    def __init__(self, cam, reader_cfg=None, motion_percent=MOTION_PERCENT):
        """cam: anything with read() -> BGR frame (vision/tictactoe/camera.py Camera)."""
        self.cam, self.reader_cfg, self.motion_percent = cam, reader_cfg, motion_percent
        self._last = self._anchor = None

    def look(self):
        frame = self.cam.read()
        small = _small(frame)
        if self._last is None:
            motion = 100.0  # nothing to compare with yet: do not trust the first frame
        else:
            # Against the frame before (fast movement) and against the frame where the
            # stillness began (slow movement: too little change from one frame to the next).
            motion = max(float(np.mean(cv2.absdiff(small, old) > PIXEL_CHANGE)) * 100
                         for old in (self._last, self._anchor))
        self._last = small
        if motion > self.motion_percent:
            self._anchor = small  # movement: the stillness starts again from here
        reading = board_reader.read_board(frame, self.reader_cfg)
        return Sight(reading.board, reading.doubtful, motion > self.motion_percent,
                     round(motion, 2), frame)

    def forget(self):
        """Call when the arm is back at the photo pose: drop the frames taken on the way.

        shortcut: drops a fixed number of frames; if old frames still show up in the lab,
        raise STALE_FRAMES or drop frames by time.
        """
        for _ in range(STALE_FRAMES):
            self.cam.read()
        self._last = self._anchor = None
