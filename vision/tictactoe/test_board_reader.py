"""Tests for the board reader and for the full chain photo -> move.

Most use fake photos (synthetic.py) to check the logic. TestRealPhotos checks
board_config.json on the real C920 photos of datasets/. Run from this folder:

    python -m unittest -v
"""

import csv
import os
import unittest

import cv2

import board_reader
import game
import robot_move
import synthetic
from game import from_text

BOARDS = ["XO./.X./..O", ".........", "XOX/OXO/OXO", "O.X/.X./X.O", "X........"]


class TestReader(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(board_reader.load_config(), **synthetic.CONFIG)

    def read(self, board, **photo_options):
        photo, corners = synthetic.make_photo(board, **photo_options)
        cfg = dict(self.cfg, board_corners_px=corners)
        return board_reader.read_board(photo, cfg)

    def test_config_skips_comments(self):
        self.assertFalse(any(k.startswith("_") for k in self.cfg))

    def test_straight_photos(self):
        for text in BOARDS:
            b = from_text(text)
            self.assertEqual(self.read(b).board, b, text)

    def test_tilted_noisy_uneven_light(self):
        for seed, text in enumerate(BOARDS):
            b = from_text(text)
            r = self.read(b, tilt=True, noise=12, light_gradient=0.35, seed=seed)
            self.assertEqual(r.board, b, text)
            self.assertEqual(r.doubtful, [], text)

    def test_cropped_photo_without_corners(self):
        b = from_text("XO./.X./..O")
        top = synthetic.draw_top_view(b)
        cfg = dict(self.cfg, board_corners_px=None)  # board_config.json has the C920 corners
        self.assertEqual(board_reader.read_board(top, cfg).board, b)

    def test_grid_shape_and_fractions(self):
        r = self.read(from_text("XO./.X./..O"))
        self.assertEqual(len(r.grid), 3)
        self.assertTrue(all(len(row) == 3 for row in r.grid))
        self.assertGreater(r.blue[0][0], r.red[0][0])
        self.assertGreater(r.red[0][1], r.blue[0][1])
        self.assertLess(r.red[2][0] + r.blue[2][0], self.cfg["min_fraction"])

    def test_empty_image(self):
        with self.assertRaises(ValueError):
            board_reader.read_board(None, self.cfg)

    def test_debug_picture(self):
        r = self.read(from_text("XO./.X./..O"))
        pic = board_reader.draw_debug(r, self.cfg)
        self.assertEqual(pic.shape[0], r.top_view.shape[0])
        self.assertEqual(pic.shape[1], 2 * r.top_view.shape[1])


class TestPhotoToMove(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(board_reader.load_config(), **synthetic.CONFIG)

    def decide(self, text, **kwargs):
        photo, corners = synthetic.make_photo(from_text(text), tilt=True, noise=8)
        cfg = dict(self.cfg, board_corners_px=corners)
        return robot_move.decide(photo, config=cfg, **kwargs)

    def test_robot_blocks(self):
        d = self.decide("XX./.O./...", robot="O")
        self.assertTrue(d.ok)
        self.assertEqual(d.cell, 3)
        self.assertEqual((d.row, d.col), (0, 2))
        self.assertEqual(d.to_line(), "MOVE;3")

    def test_waits_for_human(self):
        d = self.decide("X.O/.X./..O", robot="O")
        self.assertTrue(d.ok)
        self.assertIsNone(d.cell)
        self.assertEqual(d.to_line(), "WAIT")

    def test_game_over(self):
        d = self.decide("XXX/OO./...", robot="O")
        self.assertTrue(d.ok)
        self.assertIsNone(d.cell)
        self.assertEqual(d.to_line(), "OVER;X wins")

    def test_impossible_board_is_error(self):
        d = robot_move.decide_from_board(from_text("XXX/X../..."), robot="O")
        self.assertFalse(d.ok)
        self.assertTrue(d.to_line().startswith("ERR;"))

    def test_doubtful_reading_is_error(self):
        d = robot_move.decide_from_board(from_text("X........"), robot="O", doubtful=[5])
        self.assertFalse(d.ok)

    def test_robot_as_x_first_move(self):
        d = robot_move.decide_from_board(from_text("........."), robot="X")
        self.assertEqual(d.cell, 5)


DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
LABELS = os.path.join(DATASETS, "etiquetas_c920_2026-11-09.csv")


@unittest.skipUnless(os.path.exists(LABELS), "real photos not available")
class TestRealPhotos(unittest.TestCase):
    """The 2026-11-09 C920 photos: clean ones read right, misplaced pieces and hands are doubtful."""

    # The camera moved on 2026-11-11 (gripper remounted): these photos keep their own corners.
    CORNERS_2026_11_09 = [[579, 200], [1340, 195], [1343, 957], [584, 961]]

    def setUp(self):
        self.cfg = dict(board_reader.load_config(), board_corners_px=self.CORNERS_2026_11_09)
        with open(LABELS, encoding="utf-8") as f:
            self.rows = {row["n"]: row for row in csv.DictReader(f)}

    def read(self, n):
        path = os.path.join(DATASETS, "tictactoe_c920_2026-11-09", self.rows[n]["file"])
        return board_reader.read_board(cv2.imread(path), self.cfg)

    def test_clean_photos(self):
        for n, row in self.rows.items():
            if row["kind"] == "clean":
                r = self.read(n)
                self.assertEqual(game.to_text(r.board), row["board"], f"photo {n}")
                self.assertEqual(r.doubtful, [], f"photo {n}")

    def test_wrong_resolution_is_an_error(self):
        path = os.path.join(DATASETS, "tictactoe_c920_2026-11-09", self.rows["15"]["file"])
        small = cv2.resize(cv2.imread(path), (640, 480))
        with self.assertRaisesRegex(ValueError, "640x480 px .* 1920x1080 px"):
            board_reader.read_board(small, self.cfg)

    def test_misplaced_pieces_and_hands_are_doubtful(self):
        expected = {"5": 3, "7": 9, "9": 1, "11": 1}  # photo -> square that must be doubtful
        for n, cell in expected.items():
            self.assertIn(cell, self.read(n).doubtful, f"photo {n}")

    def test_dark_photo_is_doubtful_not_empty(self):
        self.assertEqual(self.read("16").doubtful, list(range(1, 10)))


LABELS_TODAY = os.path.join(DATASETS, "etiquetas_c920_2026-11-11.csv")


@unittest.skipUnless(os.path.exists(LABELS_TODAY), "real photos not available")
class TestCurrentPosePhotos(unittest.TestCase):
    """Photos from the photo pose in use: board_config.json's corners must fit them."""

    def test_clean_photos_with_the_config_as_is(self):
        cfg = board_reader.load_config()
        with open(LABELS_TODAY, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = os.path.join(DATASETS, "tictactoe_c920_2026-11-11", row["file"])
                r = board_reader.read_board(cv2.imread(path), cfg)
                self.assertEqual(game.to_text(r.board), row["board"], row["file"])
                self.assertEqual(r.doubtful, [], row["file"])


if __name__ == "__main__":
    unittest.main()
