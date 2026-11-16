"""Join the parts: photo -> board -> best move -> square for the robot.

This is the single entry point that the ASSAR robot code will call. We do not
know that code yet, so the interface is kept small:

    from robot_move import decide
    result = decide(photo, robot="O")
    if result.ok and result.cell:
        # tell the robot to place its piece on square result.cell (1-9)

Square numbers (seen from the camera, after straightening the board):

     1 | 2 | 3
    ---+---+---
     4 | 5 | 6
    ---+---+---
     7 | 8 | 9

The robot side must translate each square number into a pose (in URScript,
metres and radians). That table lives with the robot program, not here.

Command line:
    python robot_move.py --image photo.jpg --robot O
    python robot_move.py --board "X../.O./..X" --robot O
    python robot_move.py --play               # play against the computer in text
    python robot_move.py --play --pvp         # two players, the program is the referee
"""

import argparse
from dataclasses import dataclass, field

import game


@dataclass
class Decision:
    """Answer for the robot.

    ok        : False if the board could not be trusted (see message).
    board     : the 9 squares that were read (see game.py).
    status    : "X to move", "O to move", "X wins", "O wins" or "draw".
    cell      : square 1-9 where the robot should play, or None.
    row, col  : the same square as row and column (0-2, row 0 at the top).
    message   : short sentence for the screen or the log.
    doubtful  : squares where the colour reading was unsure.
    """
    ok: bool
    board: list
    status: str = ""
    cell: int = None
    row: int = None
    col: int = None
    message: str = ""
    doubtful: list = field(default_factory=list)

    def to_line(self):
        """One text line for a socket, e.g. "MOVE;5", "WAIT", "OVER;X wins", "ERR;...".

        A provisional format: it will change to match the ASSAR robot program.
        """
        if not self.ok:
            return f"ERR;{self.message}"
        if self.cell:
            return f"MOVE;{self.cell}"
        if self.status in ("X wins", "O wins", "draw"):
            return f"OVER;{self.status}"
        return "WAIT"


def decide_from_board(board, robot=game.O, level="hard", first_player=game.X,
                      doubtful=None, rng=None):
    """Same as decide(), but starting from a board already read (no camera)."""
    doubtful = list(doubtful or [])
    try:
        state = game.status(board, first_player)
    except game.InvalidBoard as err:
        return Decision(ok=False, board=board, message=f"Impossible board: {err}",
                        doubtful=doubtful)
    if doubtful:
        squares = ", ".join(str(c) for c in doubtful)
        return Decision(ok=False, board=board, status=state, doubtful=doubtful,
                        message=f"Unsure reading on squares {squares}: take another photo")
    if state.endswith("wins") or state == "draw":
        line = game.winning_line(board)
        extra = f" (line {line})" if line else ""
        return Decision(ok=True, board=board, status=state, message=f"Game over: {state}{extra}")
    if game.next_player(board, first_player) != robot:
        return Decision(ok=True, board=board, status=state,
                        message=f"Waiting for the human ({game.other(robot)}) to move")
    cell = game.best_move(board, robot, level=level, first_player=first_player, rng=rng)
    row, col = game.cell_to_row_col(cell)
    return Decision(ok=True, board=board, status=state, cell=cell, row=row, col=col,
                    message=f"Place {robot} on square {cell}")


def decide(image, robot=game.O, level="hard", first_player=game.X, config=None, rng=None):
    """Photo (BGR image from OpenCV) -> Decision. See the top of this file."""
    import board_reader  # here, so the text-only modes work without OpenCV
    reading = board_reader.read_board(image, config)
    return decide_from_board(reading.board, robot, level, first_player,
                             doubtful=reading.doubtful, rng=rng)


# --------------------------------------------------------------------------
# Text games on the PC (no camera, no robot)
# --------------------------------------------------------------------------

def _ask_cell(board, player):
    while True:
        text = input(f"{player}, square (1-9)? ").strip()
        if text.isdigit() and int(text) in game.empty_cells(board):
            return int(text)
        print("  That square is not free. Free squares:", game.empty_cells(board))


def play_text_safe(*args):
    """play_text(), but Ctrl+C or the end of the input just stops the game."""
    try:
        play_text(*args)
    except (KeyboardInterrupt, EOFError):
        print("\nGame stopped.")


def play_text(pvp=False, robot=game.O, level="hard", first_player=game.X):
    """Play in the console. pvp=True: two humans, the program is the referee."""
    board = [game.EMPTY] * 9
    print(game.pretty(board), "\n")
    while not game.is_over(board):
        player = game.next_player(board, first_player)
        if not pvp and player == robot:
            cell = game.best_move(board, robot, level=level, first_player=first_player)
            print(f"Computer ({robot}) plays square {cell}")
        else:
            cell = _ask_cell(board, player)
        board = game.play(board, cell, player)
        print(game.pretty(board), "\n")
    print("Result:", game.status(board, first_player))


def main():
    parser = argparse.ArgumentParser(description="Choose the robot's tic-tac-toe move.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="photo of the board")
    source.add_argument("--board", help='board as text instead of a photo, e.g. "X../.O./..X"')
    source.add_argument("--play", action="store_true", help="play a text game in the console")
    parser.add_argument("--pvp", action="store_true", help="with --play: two human players")
    parser.add_argument("--robot", default="O", choices=["X", "O"], help="robot's symbol")
    parser.add_argument("--first", default="X", choices=["X", "O"], help="who moves first")
    parser.add_argument("--level", default="hard", choices=["hard", "easy", "random"])
    parser.add_argument("--config", help="settings file for the board reader (JSON)")
    parser.add_argument("--corners", help='board corners in pixels, e.g. "150,70 500,95 530,430 120,400"')
    args = parser.parse_args()

    if args.play:
        play_text_safe(args.pvp, args.robot, args.level, args.first)
        return

    if args.board:
        result = decide_from_board(game.from_text(args.board), args.robot, args.level, args.first)
    else:
        import cv2
        import board_reader
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Could not open {args.image}")
        cfg = board_reader.load_config(args.config) if args.config else board_reader.load_config()
        if args.corners:
            cfg["board_corners_px"] = board_reader.parse_corners(args.corners)
        result = decide(image, args.robot, args.level, args.first, cfg)

    print(game.pretty(result.board))
    print()
    print(result.message)
    print("For the robot:", result.to_line())


if __name__ == "__main__":
    main()
