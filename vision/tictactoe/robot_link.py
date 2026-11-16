"""Day 4 tool: the socket between the vision PC and the robot.

The robot asks ("GET"), this program takes photos until the board is stable,
calls decide() and answers with one text line ("MOVE;5" by default). Every
detail of the conversation (who connects, port, words, message format, how
squares are numbered) is in link_config.json, so it can be changed to the
format of the ASSAR program without touching the code.

Examples (PowerShell, from vision\\tictactoe):
    # Dry run with a fixed board, no camera, no robot: two terminals
    & $PY robot_link.py --board "X../.O./..."              # terminal 1 (server)
    & $PY robot_link.py fake-robot --host 127.0.0.1        # terminal 2 (pretends to be the UR)

    # Real: camera + robot (the URScript does socket_open("<PC IP>", 50000))
    & $PY robot_link.py --camera 0 --calib ..\\..\\calib\\camera_2026-11-20.npz

Minimal URScript on the robot side (default settings):
    socket_open("192.168.1.20", 50000, "vision")
    socket_send_line("GET", "vision")
    reply = socket_read_line("vision", timeout=10)     # "MOVE;5", "WAIT", ...
With templates.move = "({index})" the robot can use
    v = socket_read_ascii_float(1, "vision")             # v[0] = count, v[1] = index
"""

import argparse
import json
import os
import socket
import sys
import time

import game
import robot_move

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LINK_CONFIG = os.path.join(HERE, "link_config.json")


def load_link_config(path=DEFAULT_LINK_CONFIG):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def format_reply(decision, cfg):
    """Decision -> the text line for the robot, using the templates of link_config.json."""
    t = cfg["templates"]
    fields = dict(cell="", index="", row="", col="", status=decision.status,
                  message=decision.message, board=game.to_text(decision.board), robot=cfg["robot"])
    if not decision.ok:
        template = t["error"]
    elif decision.cell:
        template = t["move"]
        index = cfg["cell_map"].get(str(decision.cell), decision.cell)
        fields.update(cell=decision.cell, index=index, row=decision.row, col=decision.col)
    elif decision.status in ("X wins", "O wins", "draw"):
        template = t["over"]
    else:
        template = t["wait"]
    return template.format(**fields)


class BoardSource:
    """Where the board comes from: the camera (stable reading) or a fixed text board."""

    def __init__(self, cfg, cam=None, board_text=None, reader_cfg=None):
        self.cfg, self.cam, self.reader_cfg = cfg, cam, reader_cfg
        self.board = game.from_text(board_text) if board_text else None

    def decision(self):
        c = self.cfg
        if self.board is not None:
            return robot_move.decide_from_board(self.board, c["robot"], c["level"], c["first"])
        import board_reader
        last, same = None, 0
        for _ in range(int(c["max_reads"])):
            reading = board_reader.read_board(self.cam.read(), self.reader_cfg)
            key = (tuple(reading.board), tuple(reading.doubtful))
            same = same + 1 if key == last else 1
            last = key
            if same >= int(c["stable_reads"]):
                return robot_move.decide_from_board(reading.board, c["robot"], c["level"],
                                                    c["first"], doubtful=reading.doubtful)
            time.sleep(float(c["stable_interval_s"]))
        return robot_move.Decision(ok=False, board=list(last[0]) if last else [game.EMPTY] * 9,
                                   message="Board not stable: take another photo")


def read_lines(conn, terminator):
    """Yield the text lines received on the connection (stops when it closes)."""
    buffer = ""
    while True:
        data = conn.recv(1024)
        if not data:
            return
        buffer += data.decode("utf-8", errors="replace")
        while terminator in buffer:
            line, buffer = buffer.split(terminator, 1)
            yield line.strip()


def handle(conn, source, cfg, log=print):
    """Answer the robot until it closes the connection. Returns the number of answers."""
    answers = 0
    trigger = cfg.get("trigger")
    done_words = set(cfg.get("done_words") or [])
    for line in read_lines(conn, cfg["terminator"]):
        if not line:
            continue
        if line in done_words:
            log(f"robot: {line}")
            continue
        if trigger and line != trigger:
            log(f"robot sent {line!r} (not {trigger!r}): ignored")
            continue
        decision = source.decision()
        reply = format_reply(decision, cfg)
        conn.sendall((reply + cfg["terminator"]).encode("utf-8"))
        answers += 1
        log(f"{game.to_text(decision.board)}  ->  {reply}")
    return answers


def run_server(source, cfg, once=False, log=print, ready=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((cfg["host"], int(cfg["port"])))
        srv.listen(1)
        log(f"Waiting for the robot on port {cfg['port']} (Ctrl+C to stop)...")
        if ready:
            ready.set()
        while True:
            conn, addr = srv.accept()
            log(f"Robot connected from {addr[0]}")
            with conn:
                handle(conn, source, cfg, log)
            log("Robot disconnected")
            if once:
                return


def run_client(source, cfg, log=print):
    host, port = cfg["host"], int(cfg["port"])
    log(f"Connecting to {host}:{port}...")
    with socket.create_connection((host, port), timeout=10) as conn:
        conn.settimeout(None)
        log("Connected")
        handle(conn, source, cfg, log)


def fake_robot(host, port, cfg, count=1, log=print):
    """Pretend to be the UR: send the trigger, print the answer. Returns the answers."""
    replies = []
    with socket.create_connection((host, port), timeout=10) as conn:
        lines = read_lines(conn, cfg["terminator"])
        for _ in range(count):
            conn.sendall(((cfg.get("trigger") or "GET") + cfg["terminator"]).encode("utf-8"))
            reply = next(lines)
            replies.append(reply)
            log(f"fake robot got: {reply}")
    return replies


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fake-robot":
        parser = argparse.ArgumentParser(prog="robot_link.py fake-robot",
                                         description="Pretend to be the robot (for dry runs).")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int)
        parser.add_argument("--count", type=int, default=1, help="how many moves to ask for")
        parser.add_argument("--link", default=DEFAULT_LINK_CONFIG)
        args = parser.parse_args(sys.argv[2:])
        cfg = load_link_config(args.link)
        fake_robot(args.host, args.port or int(cfg["port"]), cfg, args.count)
        return

    import camera
    parser = argparse.ArgumentParser(description="Socket between the vision PC and the robot.")
    camera.add_camera_arguments(parser)
    parser.add_argument("--board", help='fixed board instead of the camera, e.g. "X../.O./..."')
    parser.add_argument("--link", default=DEFAULT_LINK_CONFIG, help="link_config.json")
    parser.add_argument("--config", help="board_config.json for the reader")
    parser.add_argument("--role", choices=["server", "client"], help="overrides link_config.json")
    parser.add_argument("--host", help="overrides link_config.json")
    parser.add_argument("--port", type=int, help="overrides link_config.json")
    parser.add_argument("--once", action="store_true", help="server: stop after one robot connection")
    args = parser.parse_args()

    cfg = load_link_config(args.link)
    for key in ("role", "host", "port"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    if args.board:
        source = BoardSource(cfg, board_text=args.board)
    else:
        import board_reader
        reader_cfg = board_reader.load_config(args.config) if args.config else board_reader.load_config()
        if args.width is None and args.height is None and reader_cfg.get("image_size_px"):
            args.width, args.height = reader_cfg["image_size_px"]
        cam = camera.camera_from_args(args)
        print(cam.info())
        try:  # fail now, before the robot connects, not in the middle of a game
            board_reader.check_image_size(cam.read(), reader_cfg)
        except ValueError as err:
            raise SystemExit(f"Camera check: {err}")
        source = BoardSource(cfg, cam=cam, reader_cfg=reader_cfg)

    try:
        if cfg["role"] == "server":
            run_server(source, cfg, once=args.once)
        else:
            run_client(source, cfg)
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as err:
        raise SystemExit(f"Network error: {err}. Check the IP, the port and the Windows firewall.")


if __name__ == "__main__":
    main()
