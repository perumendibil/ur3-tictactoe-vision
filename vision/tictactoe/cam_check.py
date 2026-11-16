"""Day 1 tool: look at the camera, check resolution and FPS, save photos.

Keys in the window:
    s      save the current frame (to --save-dir, default ../../figuras/)
    r      turn the live board reading on/off (board, decision, MOVE;N)
    q/Esc  quit

Examples (PowerShell, from vision\\tictactoe):
    & $PY cam_check.py --camera 0
    & $PY cam_check.py --camera 1 --width 1280 --height 720 --read
    & $PY cam_check.py --camera 0 --snap 3 --save-dir ..\\..\\datasets\\tablero_2026-11-20
    & $PY cam_check.py --camera ..\\..\\datasets\\tablero_2026-11-20 --read   # photos, no camera

--snap N saves N frames without opening a window (one every --every seconds).
"""

import argparse
import os
import time

import cv2
import numpy as np

import board_reader
import camera
import robot_move

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "figuras"))


def overlay_reading(frame, cfg, robot="O", level="hard", first="X"):
    """Draw the board outline and the decision on a copy of the frame."""
    shown = frame.copy()
    corners = cfg.get("board_corners_px")
    if corners:
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(shown, [pts], True, (0, 255, 0), 2)
    try:
        result = robot_move.decide(frame, robot, level, first, cfg)
        lines = [robot_move.game.to_text(result.board), result.to_line()]
        if not result.ok:
            lines.append(result.message[:60])
    except ValueError as err:
        lines = [f"read error: {err}"]
    for i, text in enumerate(lines):
        y = 30 + 28 * i
        cv2.putText(shown, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(shown, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def main():
    parser = argparse.ArgumentParser(description="Show the camera, save photos, read the board live.")
    camera.add_camera_arguments(parser)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="folder for saved photos")
    parser.add_argument("--prefix", default=f"lab_{time.strftime('%Y-%m-%d')}_cam",
                        help="start of the photo file names")
    parser.add_argument("--read", action="store_true", help="start with the live board reading on")
    parser.add_argument("--config", default=board_reader.DEFAULT_CONFIG, help="board reader settings")
    parser.add_argument("--robot", default="O", choices=["X", "O"])
    parser.add_argument("--snap", type=int, default=0, help="save N frames and quit (no window)")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between --snap frames")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="window size, e.g. 0.5 = half (saved photos stay full size)")
    args = parser.parse_args()
    if not 0.1 <= args.scale <= 1:
        parser.error("--scale must be between 0.1 and 1")

    cam = camera.camera_from_args(args)
    print(cam.info())
    cfg = board_reader.load_config(args.config)

    if args.snap:
        for i in range(args.snap):
            frame = cam.read()
            path = camera.save_frame(frame, args.save_dir, args.prefix)
            print(f"saved {path} ({frame.shape[1]}x{frame.shape[0]} px)")
            if i < args.snap - 1:
                time.sleep(args.every)
        cam.release()
        return

    reading = args.read
    title = "cam_check (s = save, r = read board, q = quit)"
    t0, frames, fps = time.time(), 0, 0.0
    while True:
        frame = cam.read()
        frames += 1
        if time.time() - t0 >= 1.0:
            fps, t0, frames = frames / (time.time() - t0), time.time(), 0
        shown = overlay_reading(frame, cfg, args.robot) if reading else frame.copy()
        label = f"{frame.shape[1]}x{frame.shape[0]} px  {fps:.1f} FPS"
        cv2.putText(shown, label, (10, shown.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if args.scale != 1:
            shown = cv2.resize(shown, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(title, shown)
        # Photos from a folder: wait longer so they can be looked at.
        key = cv2.waitKey(500 if cam.photos else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            print("saved", camera.save_frame(frame, args.save_dir, args.prefix))
        if key == ord("r"):
            reading = not reading
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
