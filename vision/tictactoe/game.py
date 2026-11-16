"""Tic-tac-toe rules and the computer player (minimax).

The board is a list of 9 strings, read row by row from the top-left corner:

     1 | 2 | 3
    ---+---+---
     4 | 5 | 6
    ---+---+---
     7 | 8 | 9

Each square holds "X" (blue cross), "O" (red circle) or "." (empty).
Squares are numbered 1 to 9 everywhere outside this file (this is what the
robot receives). Inside the code we use list indexes 0 to 8.

No OpenCV here: this file only needs plain Python, so it can be tested on
any computer without a camera.
"""

import random

X = "X"
O = "O"
EMPTY = "."
SYMBOLS = (X, O, EMPTY)

# The 8 lines that win: 3 rows, 3 columns and 2 diagonals (list indexes).
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

# When several moves are equally good, prefer centre, then corners, then edges.
# This keeps the robot predictable (same position -> same move).
PREFERRED_ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)


class InvalidBoard(ValueError):
    """The board cannot happen in a real game (for example, two Xs too many)."""


# --------------------------------------------------------------------------
# Converting between formats
# --------------------------------------------------------------------------

def from_text(text):
    """Build a board from text such as "X.O/.X./..O" or "X.O.X...O".

    Spaces, "/" and line breaks are ignored. "-", "_" and "." mean empty.
    Lower-case letters are accepted.
    """
    cleaned = [c for c in text.upper() if c not in " /\n\r\t|"]
    board = [EMPTY if c in "-_." else c for c in cleaned]
    if len(board) != 9 or any(c not in SYMBOLS for c in board):
        raise ValueError(f"Expected 9 squares with X, O or '.', got: {text!r}")
    return board


def to_text(board):
    """One-line text version of the board, e.g. "X.O/.X./..O"."""
    rows = ["".join(board[r * 3:r * 3 + 3]) for r in range(3)]
    return "/".join(rows)


def from_grid(grid):
    """Convert a 3x3 list of rows (what the board reader returns) to a board."""
    return [cell for row in grid for cell in row]


def to_grid(board):
    """Convert a board to a 3x3 list of rows."""
    return [board[r * 3:r * 3 + 3] for r in range(3)]


def pretty(board):
    """Multi-line drawing of the board, with square numbers on empty squares."""
    shown = [c if c != EMPTY else str(i + 1) for i, c in enumerate(board)]
    rows = [" " + " | ".join(shown[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n---+---+---\n".join(rows)


def cell_to_row_col(cell):
    """Square number 1-9 -> (row, column), both 0-2, row 0 at the top."""
    if not 1 <= cell <= 9:
        raise ValueError(f"Square must be 1-9, got {cell}")
    return (cell - 1) // 3, (cell - 1) % 3


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def other(player):
    return O if player == X else X


def winner(board):
    """Return "X" or "O" if someone has three in a row, otherwise None."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def winning_line(board):
    """Return the winning squares as numbers 1-9 (e.g. (1, 5, 9)), or None."""
    for line in WIN_LINES:
        a, b, c = line
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return tuple(i + 1 for i in line)
    return None


def is_full(board):
    return EMPTY not in board


def is_over(board):
    return winner(board) is not None or is_full(board)


def empty_cells(board):
    """Free squares as numbers 1-9."""
    return [i + 1 for i, c in enumerate(board) if c == EMPTY]


def check_valid(board, first_player=X):
    """Raise InvalidBoard if this position cannot appear in a real game.

    first_player is the symbol that moves first. It is a setting because we
    do not know yet who starts in the ASSAR system.
    """
    if len(board) != 9 or any(c not in SYMBOLS for c in board):
        raise InvalidBoard("The board must have 9 squares with X, O or '.'")
    second = other(first_player)
    n_first = board.count(first_player)
    n_second = board.count(second)
    if not 0 <= n_first - n_second <= 1:
        raise InvalidBoard(
            f"{n_first} {first_player} and {n_second} {second}: "
            f"{first_player} starts, so it must have the same number or one more")
    x_wins = _has_line(board, X)
    o_wins = _has_line(board, O)
    if x_wins and o_wins:
        raise InvalidBoard("X and O cannot both have three in a row")
    # Whoever won made the last move, so the piece counts must match that.
    if (x_wins or o_wins):
        w = X if x_wins else O
        if w == first_player and n_first != n_second + 1:
            raise InvalidBoard(f"{w} has won, but then {second} played again")
        if w == second and n_first != n_second:
            raise InvalidBoard(f"{w} has won, but then {first_player} played again")


def _has_line(board, player):
    return any(board[a] == board[b] == board[c] == player for a, b, c in WIN_LINES)


def next_player(board, first_player=X):
    """Whose turn it is, worked out from the number of pieces."""
    check_valid(board, first_player)
    if board.count(first_player) == board.count(other(first_player)):
        return first_player
    return other(first_player)


def status(board, first_player=X):
    """A short summary: "X wins", "O wins", "draw", or "X to move"."""
    check_valid(board, first_player)
    w = winner(board)
    if w:
        return f"{w} wins"
    if is_full(board):
        return "draw"
    return f"{next_player(board, first_player)} to move"


def play(board, cell, player):
    """Return a NEW board with player's piece on square cell (1-9)."""
    index = cell - 1
    if not 0 <= index <= 8:
        raise ValueError(f"Square must be 1-9, got {cell}")
    if board[index] != EMPTY:
        raise ValueError(f"Square {cell} is already taken by {board[index]}")
    new_board = list(board)
    new_board[index] = player
    return new_board


def find_move(before, after):
    """Compare two photos of the board and say what move was made.

    Returns (cell, player) for exactly one new piece. Raises ValueError if
    nothing changed, more than one piece appeared, or a piece moved or
    disappeared. This is the "referee" for the two-player mode.
    """
    added = []
    for i, (b, a) in enumerate(zip(before, after)):
        if b == a:
            continue
        if b != EMPTY:
            raise ValueError(f"The piece on square {i + 1} changed or disappeared")
        added.append((i + 1, a))
    if not added:
        raise ValueError("No new piece on the board")
    if len(added) > 1:
        cells = ", ".join(str(c) for c, _ in added)
        raise ValueError(f"More than one new piece (squares {cells})")
    return added[0]


def check_turn(before, after, first_player=X):
    """Referee for two players: check that `after` is a legal move from `before`.

    Returns (cell, player). Raises ValueError with a readable reason otherwise.
    """
    if is_over(before):
        raise ValueError("The game was already over")
    expected = next_player(before, first_player)
    cell, player = find_move(before, after)
    if player != expected:
        raise ValueError(f"It was {expected}'s turn, but {player} played on square {cell}")
    return cell, player


# --------------------------------------------------------------------------
# Computer player
# --------------------------------------------------------------------------

def _toward_zero(score):
    """Move a score one step closer to 0 (a win one move later is worth less)."""
    if score > 0:
        return score - 1
    if score < 0:
        return score + 1
    return 0


def _minimax(board, to_move, me, cache):
    """Score of the position for `me` if both sides play perfectly.

    +10 means `me` has already won, -10 lost, 0 draw. Each extra move moves
    the score one step towards 0, so a quicker win scores higher and a slower
    loss scores less badly: the robot finishes the game when it can and
    resists as long as possible when it is losing.

    `cache` remembers positions already scored (there are fewer than 6000),
    so the whole game tree is solved in a fraction of a second.
    """
    key = ("".join(board), to_move)
    if key in cache:
        return cache[key]
    w = winner(board)
    if w is not None:
        score = 10 if w == me else -10
    elif is_full(board):
        score = 0
    else:
        children = []
        for i in PREFERRED_ORDER:
            if board[i] == EMPTY:
                board[i] = to_move
                children.append(_minimax(board, other(to_move), me, cache))
                board[i] = EMPTY
        best = max(children) if to_move == me else min(children)
        score = _toward_zero(best)
    cache[key] = score
    return score


def move_scores(board, player):
    """Score of every free square for `player` (bigger is better).

    Returns {cell: score}. Positive means player can force a win, 0 a draw,
    negative a loss (if the opponent plays perfectly).
    """
    cache = {}
    scores = {}
    for i in PREFERRED_ORDER:
        if board[i] == EMPTY:
            work = list(board)
            work[i] = player
            scores[i + 1] = _minimax(work, other(player), player, cache)
    return scores


def best_move(board, player=None, level="hard", easy_random=0.4, first_player=X, rng=None):
    """Choose the square (1-9) where `player` should play.

    level:
      "hard"   - perfect play with minimax: it never loses.
      "easy"   - with probability `easy_random` (0.4 = 40 %) it plays a random
                 free square; otherwise it plays the perfect move. Good for a
                 demo where the public should sometimes win.
      "random" - always a random free square (useful for testing).

    If player is None, it plays for whoever's turn it is.
    rng lets the tests fix the random numbers (random.Random(seed)).
    """
    check_valid(board, first_player)
    if is_over(board):
        raise ValueError("The game is over: there is no move to make")
    if player is None:
        player = next_player(board, first_player)
    rng = rng or random
    free = empty_cells(board)
    if level == "random" or (level == "easy" and rng.random() < easy_random):
        return rng.choice(free)
    if level not in ("hard", "easy"):
        raise ValueError(f"Unknown level {level!r}: use 'hard', 'easy' or 'random'")
    scores = move_scores(board, player)
    top = max(scores.values())
    # PREFERRED_ORDER is kept in the dict, so the first best one is the preferred one.
    return next(cell for cell, s in scores.items() if s == top)
