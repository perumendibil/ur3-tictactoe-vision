"""Tests for panel.py. Run from robot/ur:  python -m unittest -v"""

import os
import tempfile
import unittest

import cv2
import numpy as np

import panel
from eyes import Sight

import game

ROWS = """time,event,board_before,cell,store_hole,seconds,result
t,move,.../.X./...,1,0,30,ok
t,move,O.X/.X./...,7,1,20,not seen
t,over,OXX/XXO/OOX,,,,draw
t,move,.../.X./...,9,0,25,ok
t,over,.X./.XX/OOO,,,,O wins
"""


class StatsTest(unittest.TestCase):
    def test_numbers_from_the_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "partidas.csv")
            with open(path, "w") as f:
                f.write(ROWS)
            s = panel.stats(path)
        self.assertEqual(s, {"games": 2, "won": 1, "draw": 1, "lost": 0,
                             "placed": 2, "moves": 3, "s_per_move": 28,
                             "last_cell": 9, "last_s": 25.0, "last_ok": True})
        self.assertEqual(panel.stats_line(s), "2 games: robot won 1, draw 1, lost 0 | "
                                              "O placed 2/3 | 28 s per move")

    def test_no_log_yet(self):
        self.assertIn("no moves yet", panel.stats_line(panel.stats("does_not_exist.csv")))


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.sight = Sight(game.from_text("X.O......"), [5], False, 0.0,
                           np.zeros((1080, 1920, 3), np.uint8))
        self.p = panel.Panel(panel.StillEyes(self.sight), "does_not_exist.csv", window=False)

    def test_draws_every_state_and_passes_the_sight_through(self):
        p = self.p
        self.assertIs(p.look(), self.sight)
        self.assertEqual(p.render().shape, (panel.H, panel.W, 3))  # before anything is said
        p.say("ROBOT_MOVING", "Robot plays square 6: keep away")
        self.assertEqual(p.cell, 6)
        for state in panel.STATES:
            p.say(state, "some sentence long enough to be wrapped over two lines here 7")
            self.assertEqual(p.render().shape, (panel.H, panel.W, 3))
        self.assertIsNone(p.thread)  # window=False: no drawing thread

    def test_countdown_ring_restarts_when_the_number_goes_up(self):
        p = self.p
        p.say("COUNTDOWN", "Robot plays in 5 s: hands away")
        t0 = p.t_count
        p.t_count = t0 - 1.5  # as if 1.5 s had passed: the director says 4
        p.say("COUNTDOWN", "Robot plays in 4 s: hands away")
        self.assertEqual((p.t_count, p.countdown_s), (t0 - 1.5, 5))  # same countdown, kept
        p.say("COUNTDOWN", "Robot plays in 5 s: hands away")   # the board changed: restarted
        self.assertGreater(p.t_count, t0 - 1)
        p.say("MOVEMENT", "Something is moving")
        self.assertEqual((p.t_count, p.prev), (None, "COUNTDOWN"))

    def test_squares_land_inside_the_photo(self):
        cells = panel.Panel._cells_px([[579, 200], [1340, 195], [1343, 957], [584, 961]], 300)
        self.assertEqual(len(cells), 9)
        for poly in cells:
            self.assertTrue((poly[:, 0] < panel.CAM[2]).all() and (poly[:, 1] < panel.CAM[3]).all())

    def test_contact_sheet(self):
        with tempfile.TemporaryDirectory() as folder:
            path = panel.sheet(os.path.join(folder, "sheet.png"), log_path="does_not_exist.csv")
            self.assertEqual(cv2.imread(path).shape, (panel.H, 2 * panel.W, 3))


if __name__ == "__main__":
    unittest.main()
