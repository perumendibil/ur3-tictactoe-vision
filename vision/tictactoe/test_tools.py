"""Tests for the lab tools (camera, calibration, ArUco, HSV tuner, socket link).

No camera and no robot needed: they use synthetic photos and a socket on this
same computer (127.0.0.1). Run from this folder:

    python -m unittest -v
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

import cv2
import numpy as np

import board_reader
import camera
import config_tools
import game
import hsv_tuner
import robot_link
import robot_move
import synthetic

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.normpath(os.path.join(HERE, "..", "..", "calib"))
sys.path.insert(0, CALIB)
import aruco_board  # noqa: E402
import calibrate_lens  # noqa: E402


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCamera(TempDir):

    def test_folder_of_photos_loops(self):
        for i in range(2):
            cv2.imwrite(os.path.join(self.tmp, f"{i}.png"), np.full((20, 30, 3), i * 100, np.uint8))
        cam = camera.Camera(self.tmp)
        values = [int(cam.read()[0, 0, 0]) for _ in range(3)]
        self.assertEqual(values, [0, 100, 0])
        self.assertEqual(cam.last_name, "0.png")

    def test_video_file(self):
        path = os.path.join(self.tmp, "v.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        cam = camera.Camera(path)
        self.assertEqual(cam.read().shape, (48, 64, 3))
        cam.release()

    def test_save_frame_two_in_same_second(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        a = camera.save_frame(frame, self.tmp, "x")
        b = camera.save_frame(frame, self.tmp, "x")
        self.assertNotEqual(a, b)
        self.assertTrue(os.path.isfile(a) and os.path.isfile(b))

    def test_undistort_needs_same_resolution(self):
        K = np.array([[100, 0, 32], [0, 100, 24], [0, 0, 1]], float)
        und = camera.Undistorter(K, np.zeros(5), (64, 48))
        self.assertEqual(und(np.zeros((48, 64, 3), np.uint8)).shape, (48, 64, 3))
        with self.assertRaises(ValueError):
            und(np.zeros((10, 10, 3), np.uint8))

    def test_cam_check_snap(self):
        photo, _ = synthetic.make_photo(game.from_text("X........"))
        cv2.imwrite(os.path.join(self.tmp, "p.png"), photo)
        out = os.path.join(self.tmp, "out")
        subprocess.run([sys.executable, os.path.join(HERE, "cam_check.py"), "--camera", self.tmp,
                        "--snap", "2", "--every", "0", "--save-dir", out], check=True,
                       capture_output=True)
        self.assertEqual(len(os.listdir(out)), 2)


class TestConfigAndTuner(TempDir):

    def test_update_keeps_comments_and_reloads(self):
        path = os.path.join(self.tmp, "c.json")
        shutil.copy(board_reader.DEFAULT_CONFIG, path)
        config_tools.update_config_file(path, {"board_corners_px": [[1, 2], [3, 4], [5, 6], [7, 8]]})
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn('"_comment"', text)
        self.assertIn("[[1, 2], [3, 4], [5, 6], [7, 8]]", text)
        cfg = board_reader.load_config(path)
        self.assertEqual(cfg["board_corners_px"][3], [7, 8])
        self.assertEqual(cfg["warp_size_px"], 300)

    def test_tuner_round_trip(self):
        cfg = board_reader.load_config()
        back = hsv_tuner.config_from_values(hsv_tuner.values_from_config(cfg))
        self.assertEqual(back["red_ranges"], cfg["red_ranges"])
        self.assertEqual(back["blue_ranges"][0]["low"], cfg["blue_ranges"][0]["low"])
        self.assertAlmostEqual(back["min_fraction"], cfg["min_fraction"])

    def test_tuned_values_still_read_the_board(self):
        cfg = board_reader.load_config()
        live = dict(cfg, **hsv_tuner.config_from_values(hsv_tuner.values_from_config(cfg)), **synthetic.CONFIG)
        board = game.from_text("XO./.X./..O")
        photo, corners = synthetic.make_photo(board, tilt=True, noise=8, seed=3)
        self.assertEqual(board_reader.read_board(photo, dict(live, board_corners_px=corners)).board, board)


def aruco_scene(px_per_mm=2.0, tilt=True):
    """White sheet with 4 DICT_4X4_50 markers (30 mm) around a 150 mm board, seen in perspective."""
    layout_mm = {"0": [-40, -40], "1": [190, -40], "2": [190, 190], "3": [-40, 190], "4": [75, -40]}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side_mm, offset_mm = 30, 80  # sheet origin is 80 mm left/up of the board origin
    size = int(380 * px_per_mm)
    sheet = np.full((size, size), 255, np.uint8)
    for key, (x, y) in layout_mm.items():
        s = int(side_mm * px_per_mm)
        marker = cv2.aruco.generateImageMarker(dictionary, int(key), s)
        cx, cy = (x + offset_mm) * px_per_mm, (y + offset_mm) * px_per_mm
        sheet[int(cy - s / 2):int(cy - s / 2) + s, int(cx - s / 2):int(cx - s / 2) + s] = marker
    # mm (board frame) -> sheet px
    A = np.array([[px_per_mm, 0, offset_mm * px_per_mm], [0, px_per_mm, offset_mm * px_per_mm], [0, 0, 1]])
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = np.float32([[60, 40], [size + 20, 70], [size, size + 30], [40, size + 10]]) if tilt else src + 30
    P = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(sheet, P, (size + 80, size + 80), borderValue=255)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), layout_mm, P @ A  # last: mm -> image px


class TestAruco(TempDir):

    def test_diagonal_centre_of_a_square(self):
        c = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        np.testing.assert_allclose(aruco_board.diagonal_centre(c), [5, 5])

    def test_homography_and_board_corners(self):
        image, layout, mm_to_px = aruco_scene()
        found = aruco_board.detect(image, "DICT_4X4_50")
        self.assertEqual(sorted(found), [0, 1, 2, 3, 4])
        H, errors = aruco_board.homography(found, layout)
        self.assertLess(max(errors.values()), 0.5)  # mm
        corners_mm = [[0, 0], [150, 0], [150, 150], [0, 150]]
        got = np.array(aruco_board.board_corners_px(H, corners_mm), float)
        expected = cv2.perspectiveTransform(np.float64(corners_mm).reshape(-1, 1, 2), mm_to_px).reshape(-1, 2)
        self.assertLess(np.abs(got - expected).max(), 2.0)  # px

    def test_cli_writes_corners_into_config(self):
        image, layout, _ = aruco_scene()
        img_path = os.path.join(self.tmp, "a.png")
        cv2.imwrite(img_path, image)
        layout_path = os.path.join(self.tmp, "layout.json")
        with open(layout_path, "w") as f:
            json.dump({"markers_mm": layout, "board_corners_mm": [[0, 0], [150, 0], [150, 150], [0, 150]]}, f)
        cfg_path = os.path.join(self.tmp, "board_config.json")
        shutil.copy(board_reader.DEFAULT_CONFIG, cfg_path)
        h_path = os.path.join(self.tmp, "H.npy")
        subprocess.run([sys.executable, os.path.join(CALIB, "aruco_board.py"), "--image", img_path,
                        "--layout", layout_path, "--save-h", h_path, "--write-config",
                        "--config", cfg_path], check=True, capture_output=True)
        self.assertEqual(np.load(h_path).shape, (3, 3))
        self.assertEqual(len(board_reader.load_config(cfg_path)["board_corners_px"]), 4)


def chessboard_views(K, cols=9, rows=6, square_mm=25, n=12, size=(640, 480)):
    """Synthetic photos of a chessboard seen by a camera with matrix K and no distortion."""
    sq_px = 20  # px per square in the flat board picture
    w, h = (cols + 1) * sq_px, (rows + 1) * sq_px
    flat = np.zeros((h + 2 * sq_px, w + 2 * sq_px), np.uint8) + 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = sq_px + r * sq_px, sq_px + c * sq_px
                flat[y:y + sq_px, x:x + sq_px] = 0
    # flat px -> board mm (origin at the first inner corner)
    S = np.array([[square_mm / sq_px, 0, -2 * square_mm], [0, square_mm / sq_px, -2 * square_mm], [0, 0, 1]])
    rng = np.random.default_rng(1)
    images = []
    for i in range(n):
        rvec = rng.uniform(-0.35, 0.35, 3)
        rvec[2] = rng.uniform(-0.3, 0.3)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-60, 0), rng.uniform(-50, 0), rng.uniform(450, 600)])
        Hmm2px = K @ np.column_stack([R[:, 0], R[:, 1], t])
        Hm = Hmm2px @ S
        img = cv2.warpPerspective(flat, Hm / Hm[2, 2], size, borderValue=255)
        images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return images


class TestLensCalibration(TempDir):

    def test_recovers_focal_length(self):
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        for i, img in enumerate(chessboard_views(K)):
            cv2.imwrite(os.path.join(self.tmp, f"{i:02d}.png"), img)
        paths = calibrate_lens.list_photos(self.tmp)
        result = calibrate_lens.calibrate(paths, 9, 6, 25)
        self.assertGreaterEqual(len(result["paths"]), 8)
        self.assertLess(result["rms"], 0.5)  # px
        self.assertAlmostEqual(result["K"][0, 0] / 600.0, 1.0, delta=0.03)
        found, errors = calibrate_lens.check(paths, 9, 6, 25, result["K"], result["dist"])
        self.assertLess(np.mean(errors), 0.5)

    def test_verdict(self):
        self.assertEqual(calibrate_lens.verdict(0.3), "very good")
        self.assertEqual(calibrate_lens.verdict(0.8), "acceptable")


class TestRobotLink(TempDir):

    def setUp(self):
        super().setUp()
        self.cfg = robot_link.load_link_config()

    def test_default_reply_is_move_n(self):
        d = robot_move.decide_from_board(game.from_text("X../.../..."), "O")
        self.assertEqual(robot_link.format_reply(d, self.cfg), f"MOVE;{d.cell}")

    def test_custom_template_and_zero_based_cells(self):
        cfg = dict(self.cfg, templates=dict(self.cfg["templates"], move="({index})"),
                   cell_map={str(i): i - 1 for i in range(1, 10)})
        d = robot_move.decide_from_board(game.from_text("X../.../..."), "O")
        self.assertEqual(robot_link.format_reply(d, cfg), f"({d.cell - 1})")

    def test_wait_over_error(self):
        f = lambda text: robot_link.format_reply(robot_move.decide_from_board(game.from_text(text), "O"), self.cfg)
        self.assertEqual(f("........."), "WAIT")          # X starts: the human moves first
        self.assertEqual(f("XXX/OO./..."), "OVER;X wins")
        self.assertTrue(f("XXX/XX./...").startswith("ERR;"))

    def _serve(self, source, cfg):
        with socket.socket() as s:  # find a free port
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cfg = dict(cfg, host="127.0.0.1", port=port)
        ready = threading.Event()
        thread = threading.Thread(target=robot_link.run_server,
                                  args=(source, cfg), kwargs=dict(once=True, log=lambda *_: None, ready=ready),
                                  daemon=True)
        thread.start()
        ready.wait(5)
        return cfg, thread

    def test_server_and_fake_robot_over_a_real_socket(self):
        source = robot_link.BoardSource(self.cfg, board_text="X../.../...")
        cfg, thread = self._serve(source, self.cfg)
        replies = robot_link.fake_robot("127.0.0.1", cfg["port"], cfg, count=2, log=lambda *_: None)
        thread.join(5)
        self.assertEqual(len(replies), 2)
        self.assertTrue(all(r.startswith("MOVE;") for r in replies))

    def test_ignores_other_lines_and_done(self):
        source = robot_link.BoardSource(self.cfg, board_text="X../.../...")
        cfg, thread = self._serve(source, self.cfg)
        with socket.create_connection(("127.0.0.1", cfg["port"]), timeout=5) as conn:
            conn.sendall(b"HELLO\nDONE\nGET\n")
            reply = next(robot_link.read_lines(conn, "\n"))
        thread.join(5)
        self.assertTrue(reply.startswith("MOVE;"))

    def test_stable_reading_from_photos(self):
        board = game.from_text("X../.O./..X")
        photo, corners = synthetic.make_photo(board, tilt=True, noise=5, seed=2)
        cv2.imwrite(os.path.join(self.tmp, "p.png"), photo)
        reader_cfg = dict(board_reader.load_config(), board_corners_px=corners, **synthetic.CONFIG)
        cfg = dict(self.cfg, stable_interval_s=0)
        source = robot_link.BoardSource(cfg, cam=camera.Camera(self.tmp), reader_cfg=reader_cfg)
        d = source.decision()
        self.assertTrue(d.ok)
        self.assertEqual(d.board, board)
        self.assertIsNotNone(d.cell)


if __name__ == "__main__":
    unittest.main()
