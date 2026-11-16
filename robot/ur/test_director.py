"""Tests for director.py with a fake camera, arm and gripper (nothing real is needed).

Run from robot/ur:  python -m unittest -v
"""

import contextlib
import io
import unittest

from announce import Announcer
from director import VERIFY_S, Director, DryArm
from eyes import Sight
from ur_motion import load_positions

import game

POS = load_positions()


class Rig:
    """A director with fakes. Each camera frame takes 1 s of the fake clock."""

    def __init__(self, level="hard", countdown_s=5):
        self.frames, self.now, self.did, self.said, self.rows, self.asked = [], 0.0, [], [], [], []
        self.director = Director(self, self, self, POS,
                                 Announcer(lambda state, text: self.said.append((state, text))),
                                 level=level, countdown_s=countdown_s, clock=lambda: self.now,
                                 ask=self.asked.append, log=lambda *row: self.rows.append(row))

    def see(self, text, times=1, moving=False, doubtful=()):
        """Queue frames, then run one director step per frame still in the queue."""
        self.frames += [Sight(game.from_text(text), list(doubtful), moving, 0.0, None)] * times
        while self.frames:
            self.director.step()

    @property
    def states(self):
        return [state for state, _ in self.said]

    # the eyes
    def look(self):
        self.now += 1
        if not self.frames:
            return Sight([game.EMPTY] * 9, [], True, 100.0, None)  # nothing queued: never clear
        return self.frames.pop(0)

    def forget(self):
        self.did.append("forget")

    # the arm and the gripper
    def open(self):
        self.did.append("open")

    def close(self):
        self.did.append("close")

    def pick(self, pos, hole):
        self.did.append(f"pick {hole}")

    def carry(self, pos, hole, cell):
        self.did.append(f"carry {hole}->{cell}")

    def go_photo(self, pos):
        self.did.append("photo")


TURN = ["open", "pick 0", "close", "carry 0->5", "open", "photo", "forget"]


class DirectorTest(unittest.TestCase):
    def test_waits_for_the_person(self):
        rig = Rig()
        rig.see(".../.../...", 3)
        rig.see("X../.O./...", 20)
        self.assertEqual(rig.did, [])
        self.assertEqual(rig.states, ["YOUR_TURN", "YOUR_TURN"])

    def test_robot_plays_after_the_countdown_and_checks_its_piece(self):
        rig = Rig()
        rig.see("X../.../...", 5)
        self.assertEqual(rig.did, [])  # 5 frames = seconds 5, 4, 3, 2, 1 on the screen
        self.assertEqual([t for s, t in rig.said if s == "COUNTDOWN"],
                         [f"Robot plays in {n} s: hands away" for n in (5, 4, 3, 2, 1)])
        rig.frames += [Sight(game.from_text("X../.../..."), [], False, 0.0, None)]
        rig.see("X../.O./...", 3)  # the sixth frame starts the move; then the camera shows the O
        self.assertEqual(rig.did, TURN)  # hard level: centre
        self.assertEqual(rig.asked, [])
        self.assertEqual(rig.rows, [("move", "X../.../...", 5, 0, 0.0, "ok")])
        self.assertEqual(game.to_text(rig.director.before), "X../.O./...")

    def test_movement_or_a_new_board_restart_the_countdown(self):
        for interruption in (dict(text="X../.../...", moving=True),
                             dict(text="X../.../...", doubtful=[5]),
                             dict(text=".X./.../...")):
            rig = Rig()
            rig.see("X../.../...", 4)
            rig.see(**interruption)
            rig.see("X../.../...", 5)
            self.assertEqual(rig.did, [], interruption)

    def test_store_hole_is_the_number_of_os_on_the_board(self):
        rig = Rig()
        rig.frames += [Sight(game.from_text("XXO/.O./X.."), [], False, 0.0, None)] * 6
        rig.see("XXO/.O./X..", 4)
        self.assertEqual(rig.did[1:4], ["pick 2", "close", "carry 2->4"])

    def test_piece_that_did_not_arrive_pauses_and_asks(self):
        rig = Rig()
        rig.see("X../.../...", 6)  # after the move the camera never shows the O
        self.assertEqual(rig.did, TURN + ["forget"])
        self.assertEqual(len(rig.asked), 1)
        self.assertEqual(rig.states[-1], "PAUSED")
        self.assertEqual(rig.rows[0][-1], "not seen")
        self.assertIsNone(rig.director.before)
        self.assertGreaterEqual(rig.now, 6 + VERIFY_S)

    def test_a_hand_during_the_check_does_not_count(self):
        rig = Rig()
        good = Sight(game.from_text("X../.O./..."), [], False, 0.0, None)
        hand = Sight(game.from_text("X../.O./..."), [], True, 50.0, None)
        rig.frames += [Sight(game.from_text("X../.../..."), [], False, 0.0, None)] * 6
        rig.frames += [good, good, hand, good, good, good]
        rig.see("X../.../...", 0)
        self.assertEqual(rig.asked, [])
        self.assertEqual(rig.frames, [])  # it needed the three good frames after the hand

    def test_game_over_waits_for_an_empty_board(self):
        rig = Rig()
        rig.see("XXX/OO./...", 3)
        self.assertEqual(rig.rows, [("over", "XXX/OO./...", "", "", "", "X wins")])
        rig.see("X../.../...", 20)  # clearing the board: looks like "O to move", must not play
        self.assertEqual(rig.did, [])
        rig.see(".../.../...")
        self.assertEqual(rig.states, ["GAME_OVER", "YOUR_TURN"])
        rig.see("X../.../...", 6)
        self.assertEqual(rig.did[:2], ["open", "pick 0"])

    def test_referee_stops_a_moved_piece(self):
        rig = Rig()
        rig.director.before = game.from_text("X../.O./...")
        rig.see(".XX/.O./...", 20)  # the X of square 1 moved and a new one appeared
        self.assertEqual(rig.did, [])
        self.assertEqual(rig.states, ["ILLEGAL"])

    def test_impossible_board_never_moves(self):
        rig = Rig()
        rig.see("XXX/.../...", 20)
        self.assertEqual(rig.did, [])
        self.assertEqual(rig.states, ["BAD_BOARD"])

    def test_dry_arm_prints_the_urscript(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            DryArm().pick(POS, 0)
            DryArm().carry(POS, 0, 5)
            DryArm().go_photo(POS)
        self.assertEqual(out.getvalue().count("movel("), 3 + 5 + 1)
        self.assertIn("movej(", out.getvalue())


if __name__ == "__main__":
    unittest.main()
