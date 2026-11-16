"""Tests for eyes.py and announce.py with the lab photos of 2026-11-10 (no camera).

Run from robot/ur:  python -m unittest -v
"""

import os
import unittest

import cv2

from announce import Announcer
from eyes import STALE_FRAMES, Eyes

import board_reader  # noqa: E402  (vision/tictactoe)

PHOTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "datasets", "tictactoe_c920_2026-11-10")
# The camera moved on 2026-11-11 (gripper remounted): these photos keep their own corners.
CFG = dict(board_reader.load_config(), board_corners_px=[[579, 200], [1340, 195], [1343, 957], [584, 961]])


def make_eyes(cam):
    return Eyes(cam, CFG)


def photo(name):
    image = cv2.imread(os.path.join(PHOTOS, name))
    assert image is not None, f"missing lab photo {name}"
    return image


CLEAN = photo("lab_2026-11-10_cam_2026-11-10_17-12-27.png")        # ..O / XX. / ...
PUSHED = photo("lab_2026-11-10_cam_2026-11-10_17-13-39.png")       # the X of square 5 pushed aside
HAND = photo("lab_2026-11-10_cam_2026-11-10_17-14-10.png")         # a hand at the corner
BIG_HAND = photo("mano_lee_9_O_2026-11-10_19-14-46.png")           # a hand over the board


class Frames:
    """A camera that gives a fixed list of frames."""

    def __init__(self, *frames):
        self.frames, self.reads = list(frames), 0

    def read(self):
        self.reads += 1
        return self.frames.pop(0)


class EyesTest(unittest.TestCase):
    def test_first_frame_is_never_trusted(self):
        sight = make_eyes(Frames(CLEAN)).look()
        self.assertTrue(sight.moving)
        self.assertFalse(sight.clear)

    def test_still_board_is_read_and_clear(self):
        eyes = make_eyes(Frames(CLEAN, CLEAN))
        eyes.look()
        sight = eyes.look()
        self.assertEqual("".join(sight.board), "..O" "XX." "...")
        self.assertEqual(sight.motion_percent, 0)
        self.assertTrue(sight.clear)

    def test_one_piece_pushed_aside_is_not_trusted(self):
        # 1.5 % change: below the movement limit, but the half-covered square is doubtful.
        eyes = make_eyes(Frames(CLEAN, PUSHED))
        eyes.look()
        sight = eyes.look()
        self.assertEqual(sight.doubtful, [5])
        self.assertFalse(sight.clear)

    def test_a_hand_is_movement_and_doubt(self):
        for hand in (HAND, BIG_HAND):
            eyes = make_eyes(Frames(CLEAN, hand, hand))
            eyes.look()
            self.assertTrue(eyes.look().moving)
            still_hand = eyes.look()  # a hand that stays still: no movement, but doubtful
            self.assertFalse(still_hand.moving)
            self.assertFalse(still_hand.clear)

    def test_slow_movement_is_seen(self):
        # A hand-sized patch creeping over the O store, 2 px per frame (1920 px wide photo).
        def creeping(step):
            frame = CLEAN.copy()
            cv2.rectangle(frame, (1500 + 2 * step, 400), (1900 + 2 * step, 1080), (150, 170, 220), -1)
            return frame
        eyes = make_eyes(Frames(*[creeping(s) for s in range(100)]))
        sights = [eyes.look() for _ in range(100)]
        # Second frame: compared only with the first one, so this is the frame-to-frame change.
        self.assertFalse(sights[1].moving, "too fast: this test must creep below the limit")
        self.assertTrue(any(s.moving for s in sights[2:]))

    def test_forget_drops_old_frames_and_distrusts_the_next_one(self):
        cam = Frames(CLEAN, *[HAND] * STALE_FRAMES, CLEAN, CLEAN)
        eyes = Eyes(cam, CFG)
        eyes.look()
        eyes.forget()
        self.assertEqual(cam.reads, 1 + STALE_FRAMES)
        self.assertTrue(eyes.look().moving)
        self.assertTrue(eyes.look().clear)


class AnnouncerTest(unittest.TestCase):
    def test_every_output_gets_the_message_once(self):
        screen, lcd = [], []
        announcer = Announcer(lambda s, t: screen.append((s, t)), lambda s, t: lcd.append((s, t)))
        announcer.say("COUNTDOWN", "5")
        announcer.say("COUNTDOWN", "5")
        announcer.say("COUNTDOWN", "4")
        self.assertEqual(screen, [("COUNTDOWN", "5"), ("COUNTDOWN", "4")])
        self.assertEqual(lcd, screen)

    def test_a_broken_output_does_not_stop_the_others(self):
        screen = []
        announcer = Announcer(lambda s, t: 1 / 0, lambda s, t: screen.append(s))
        announcer.say("ERROR", "x")
        self.assertEqual(screen, ["ERROR"])


if __name__ == "__main__":
    unittest.main()
