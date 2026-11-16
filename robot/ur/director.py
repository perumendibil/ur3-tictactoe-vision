"""The director: the PC program that plays tic-tac-toe with the UR3 against a person.

It joins the eyes (camera + board reader), the game rules, the arm, the gripper
and the announcer. Decisions of 2026-11-10 (DECISIONES.md): the robot is O, the
person is X and starts; the robot moves after ONE countdown that starts again if
anything moves; the photo of the board is the truth, so the program can be
stopped and started again in the middle of a game.

    python director.py --seco             # dry run: real camera, the arm and gripper only print
    python director.py                    # real robot (Peru's laptop: camera 1, COM3)
    python director.py --level hard       # the robot never loses

States (one camera frame = one step):

    PLAYING ----- three in a row, or board full -----> GAME_OVER
       ^                                                   |
       +------------- board empty and still ---------------+

Inside PLAYING the photo decides: X to move -> wait for the person; O to move ->
countdown -> robot turn (open, pick, close, carry, open, photo pose) -> check
with the camera that the O is on its square. If it is not: stop and ask a person.

Limit to remember: the camera is on the wrist, so nothing watches the board
while the arm moves. Hands away after the countdown.
"""

import argparse
import csv
import math
import os
import time

from announce import Announcer, console
from eyes import Eyes                      # also adds vision/tictactoe to the import path
from panel import Panel
from gripper import Gripper, GripperError
from ur_motion import UR, RobotError, carry_program, load_positions, photo_program, pick_program

import board_reader  # noqa: E402  (vision/tictactoe)
import game          # noqa: E402
from camera import Camera  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HUMAN, ROBOT = game.X, game.O
VERIFY_FRAMES = 3   # clear frames in a row that must show the new O on its square
VERIFY_S = 10       # s to get them after the arm is back at the photo pose
LOG_COLUMNS = ("time", "event", "board_before", "cell", "store_hole", "seconds", "result")


class Director:
    def __init__(self, eyes, arm, hand, pos, announcer, level="easy", countdown_s=5,
                 clock=time.monotonic, ask=input, log=lambda *row: None, rng=None):
        self.eyes, self.arm, self.hand, self.pos = eyes, arm, hand, pos
        self.say, self.level, self.countdown_s = announcer.say, level, countdown_s
        self.clock, self.ask, self.log, self.rng = clock, ask, log, rng
        self.state = "PLAYING"
        self.before = None       # last board with X to move that we are sure of (for the referee)
        self.count_board = None  # the countdown runs for this board...
        self.count_from = None   # ...since this time

    def step(self):
        """Look at one camera frame and do what it asks for."""
        sight = self.eyes.look()
        if not sight.clear:
            self.count_from = None
            if sight.moving:
                self.say("MOVEMENT", "Something is moving: waiting for a still board")
            else:
                self.say("UNSURE", f"I cannot read square(s) {sight.doubtful}: "
                                   "straighten the piece or take the hand away")
            return
        board = sight.board
        try:
            status = game.status(board, HUMAN)
        except game.InvalidBoard as err:
            self.count_from = None
            self.say("BAD_BOARD", f"Impossible board: {err}")
            return

        if self.state == "GAME_OVER":
            if board.count(game.EMPTY) < 9:
                return  # still clearing the board: never play in the middle of that
            self.state = "PLAYING"
        if status.endswith("wins") or status == "draw":
            self.state, self.before, self.count_from = "GAME_OVER", None, None
            self.log("over", game.to_text(board), "", "", "", status)
            self.say("GAME_OVER", f"{status}. Clear the board and refill the O store to play again")
        elif status.startswith(HUMAN):
            self.count_from = None
            if board.count(game.EMPTY) == 9:
                self.before = board
                self.say("YOUR_TURN", "New game: you are X, place a piece (O store full?)")
            else:
                self.say("YOUR_TURN", "Your turn (X)")
        else:
            self._robot_to_move(board)

    def _robot_to_move(self, board):
        if self.before is not None:
            try:
                game.check_turn(self.before, board, HUMAN)
            except ValueError as err:
                self.count_from = None
                self.say("ILLEGAL", f"{err}. Put the board back as it was")
                return
        now = self.clock()
        if self.count_from is None or board != self.count_board:
            self.count_board, self.count_from = board, now
        left = self.countdown_s - (now - self.count_from)
        if left > 0:
            self.say("COUNTDOWN", f"Robot plays in {math.ceil(left)} s: hands away")
            return
        self.count_from = None
        self._robot_turn(board)

    def _robot_turn(self, board):
        cell = game.best_move(board, ROBOT, level=self.level, first_player=HUMAN, rng=self.rng)
        # shortcut: takes store hole number "O pieces on the board", so the store must be
        # refilled in order; upgrade: read the free holes from the photo (the store is in view).
        hole = board.count(ROBOT)
        self.say("ROBOT_MOVING", f"Robot plays square {cell}: keep away")
        start = self.clock()
        self.hand.open()
        self.arm.pick(self.pos, hole)
        self.hand.close()
        self.arm.carry(self.pos, hole, cell)
        self.hand.open()
        self.arm.go_photo(self.pos)
        seconds = round(self.clock() - start, 1)

        # The gripper cannot feel a block: only the camera knows if the O arrived.
        self.eyes.forget()
        seen, deadline = 0, self.clock() + VERIFY_S
        while seen < VERIFY_FRAMES and self.clock() < deadline:
            sight = self.eyes.look()
            seen = seen + 1 if sight.clear and sight.board[cell - 1] == ROBOT else 0
        arrived = seen >= VERIFY_FRAMES
        self.log("move", game.to_text(board), cell, hole, seconds, "ok" if arrived else "not seen")
        if arrived:
            self.before = game.play(board, cell, ROBOT)
            return
        self.say("PAUSED", f"I cannot see my O on square {cell}. Put it there by hand "
                           "and check the O store")
        self.ask("Press Enter when the board is right... ")
        self.before = None  # a person touched the board: trust the next photo
        self.eyes.forget()


# --------------------------------------------------------------------------
# Dry run and command line
# --------------------------------------------------------------------------

class DryArm:
    """--seco: prints the URScript the real arm would get. Nothing moves."""

    def pick(self, pos, hole):
        # As if starting from the photo pose: above the travel height.
        print(f"[dry] pick from store hole {hole}:\n{pick_program(pos, hole, (0, 0, 1000))}")

    def carry(self, pos, hole, cell):
        print(f"[dry] carry to square {cell}:\n{carry_program(pos, hole, cell)}")

    def go_photo(self, pos, lift=True):
        print(f"[dry] photo pose:\n{photo_program(pos, lift)}")

    def stop(self):
        pass


class DryHand:
    def open(self):
        print("[dry] gripper opens")

    def close(self):
        print("[dry] gripper closes")

    def release_port(self):
        pass


def csv_log(path):
    def write(*row):
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(LOG_COLUMNS)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), *row])
    return write


def prepare(arm, pos, port):
    """Check the arm, open the gripper and bring the arm to the photo pose. Asks before moving."""
    if not arm.ready():
        raise SystemExit("Robot not ready: " + arm.dashboard("robotmode") + " | "
                         + arm.dashboard("safetymode") + " (release the brakes on the pendant)")
    hand = Gripper(port, pos["gripper_open_set"], pos["gripper_close_set"])
    # Always, not only when inactive: squeezing a block makes the weak servo's gears slip,
    # and then the same SET closes more or less; the full sweep puts them back (2026-11-11).
    answer = input("Calibrating the gripper sweeps the servo: gripper EMPTY and hands away. "
                   "Type YES to calibrate: ")
    if answer.strip().upper() != "YES":
        raise SystemExit("Gripper not calibrated: nothing was moved.")
    hand.activate()
    input("The arm will go to the photo pose. Check: gripper EMPTY (a piece it holds goes back "
          "to the store), O store filled in order, pendant speed low the first time, hand on "
          "the emergency stop. Press Enter to move... ")
    # Already above the travel height (e.g. at the photo pose): do not come down first.
    arm.go_photo(pos, lift=arm.state()[1][2] < pos["travel_z_mm"])
    return hand


def main():
    parser = argparse.ArgumentParser(description="Play tic-tac-toe with the UR3 against a person.")
    parser.add_argument("--seco", action="store_true",
                        help="dry run: real camera, but the arm and the gripper only print")
    parser.add_argument("--camera", default="1",
                        help="camera index (1 on Peru's laptop), or a photo / folder of photos")
    parser.add_argument("--port", default="COM3", help="serial port of the gripper Arduino")
    parser.add_argument("--level", default="easy", choices=["easy", "hard"])
    parser.add_argument("--countdown", type=float, default=5, help="seconds before the robot moves")
    parser.add_argument("--log", default=os.path.join(HERE, "partidas.csv"),
                        help="one line per robot move (not written with --seco)")
    args = parser.parse_args()

    pos = load_positions()
    cam = Camera(args.camera, width=1920, height=1080)
    panel = Panel(Eyes(cam, board_reader.load_config()), args.log)  # window for the pitch video
    announcer = Announcer(console, panel.say)
    # The arm exists before anything moves, so the stop below always reaches the real robot.
    arm, hand = (DryArm() if args.seco else UR(pos["robot_ip"])), DryHand()
    try:
        if not args.seco:
            hand = prepare(arm, pos, args.port)
        director = Director(panel, arm, hand, pos, announcer,
                            level=args.level, countdown_s=args.countdown,
                            log=(lambda *row: None) if args.seco else csv_log(args.log))
        while True:
            director.step()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped from the keyboard.")
    except (RobotError, GripperError, OSError) as err:
        announcer.say("ERROR", f"{err}. Program stopped: fix it and start again")
    finally:
        try:
            arm.stop()  # whatever happened, the arm must not keep moving on its own
        except OSError as err:
            print(f"COULD NOT SEND THE STOP TO THE ROBOT ({err}): use the pendant")
        hand.release_port()
        cam.release()


if __name__ == "__main__":
    main()
