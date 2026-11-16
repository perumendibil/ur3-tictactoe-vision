"""Day 3 tool: tune the red and blue HSV ranges of board_config.json live.

A window with sliders (trackbars) and the board reader's debug picture:
left the straightened board with what was read in each square, right the red
and blue masks. Move the sliders until each piece is fully coloured in the
mask and the background is black.

Keys:  s = save the values into board_config.json   n = next photo (folders)
       p = print the per-square red/blue shares       q/Esc = quit

Examples (PowerShell, from vision\\tictactoe):
    & $PY hsv_tuner.py --camera 0
    & $PY hsv_tuner.py --camera ..\\..\\datasets\\tablero_2026-11-20     # folder of photos

Hue in OpenCV goes 0-179. Red sits at both ends, so it has two bands:
0..R_H_low_max and R_H_high_min..179. Run board_reader.py --pick first, so
board_corners_px is set; without it the whole photo is taken as the board.
"""

import argparse

import cv2

import board_reader
import camera
import config_tools

# (slider name, max value). Red has two hue bands; S and V share the minimum.
SLIDERS = [
    ("R H low max", 179), ("R H high min", 179), ("R S min", 255), ("R V min", 255),
    ("B H min", 179), ("B H max", 179), ("B S min", 255), ("B V min", 255),
    ("min fraction %", 50),
]


def values_from_config(cfg):
    """board_config values -> slider positions."""
    red, blue = cfg["red_ranges"], cfg["blue_ranges"][0]
    low_band = min(red, key=lambda r: r["low"][0])
    high_band = max(red, key=lambda r: r["low"][0])
    return {
        "R H low max": low_band["high"][0], "R H high min": high_band["low"][0],
        "R S min": low_band["low"][1], "R V min": low_band["low"][2],
        "B H min": blue["low"][0], "B H max": blue["high"][0],
        "B S min": blue["low"][1], "B V min": blue["low"][2],
        "min fraction %": int(round(100 * float(cfg["min_fraction"]))),
    }


def config_from_values(v):
    """Slider positions -> the keys of board_config.json that they control."""
    rs, rv = v["R S min"], v["R V min"]
    return {
        "red_ranges": [
            {"low": [0, rs, rv], "high": [v["R H low max"], 255, 255]},
            {"low": [v["R H high min"], rs, rv], "high": [179, 255, 255]},
        ],
        "blue_ranges": [
            {"low": [v["B H min"], v["B S min"], v["B V min"]], "high": [v["B H max"], 255, 255]},
        ],
        "min_fraction": max(1, v["min fraction %"]) / 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune the HSV colour ranges of the board reader.")
    camera.add_camera_arguments(parser)
    parser.add_argument("--config", default=board_reader.DEFAULT_CONFIG, help="board_config.json")
    args = parser.parse_args()

    cfg = board_reader.load_config(args.config)
    cam = camera.camera_from_args(args)
    print(cam.info())
    if not cfg.get("board_corners_px"):
        print("board_corners_px is not set: the whole photo is used as the board "
              "(run board_reader.py --pick first)")

    controls = "HSV sliders"
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls, 420, 380)
    for name, top in SLIDERS:
        cv2.createTrackbar(name, controls, 0, top, lambda _v: None)
    for name, value in values_from_config(cfg).items():
        cv2.setTrackbarPos(name, controls, int(value))

    frame = cam.read()
    while True:
        values = {name: cv2.getTrackbarPos(name, controls) for name, _ in SLIDERS}
        live = dict(cfg, **config_from_values(values))
        reading = board_reader.read_board(frame, live)
        debug = board_reader.draw_debug(reading, live)
        cv2.imshow("board (left) and masks (right)", cv2.resize(debug, None, fx=1.5, fy=1.5))
        key = cv2.waitKey(30) & 0xFF
        if cam.cap is not None:
            frame = cam.read()  # live camera: always the newest frame
        if key in (ord("q"), 27):
            break
        if key == ord("n") and cam.photos:
            frame = cam.read()
            print("photo:", cam.last_name)
        if key == ord("p"):
            board_reader.print_reading(reading)
        if key == ord("s"):
            config_tools.update_config_file(args.config, config_from_values(values))
            print(f"Saved to {args.config}:", config_from_values(values))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
