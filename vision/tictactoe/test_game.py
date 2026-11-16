"""Tests for the game rules and the computer player. Run from this folder:

    python -m unittest -v
"""

import random
import unittest

import game
from game import EMPTY, O, X, from_text


class TestRules(unittest.TestCase):

    def test_text_round_trip(self):
        b = from_text("XO./.X./..O")
        self.assertEqual(game.to_text(b), "XO./.X./..O")
        self.assertEqual(from_text("xo-  -x-  --o"), b)
        self.assertEqual(game.from_grid(game.to_grid(b)), b)

    def test_bad_text(self):
        with self.assertRaises(ValueError):
            from_text("XO")
        with self.assertRaises(ValueError):
            from_text("XOZ......")

    def test_all_eight_lines_win(self):
        for line in game.WIN_LINES:
            b = [EMPTY] * 9
            for i in line:
                b[i] = X
            self.assertEqual(game.winner(b), X)
            self.assertEqual(game.winning_line(b), tuple(i + 1 for i in line))

    def test_no_winner(self):
        self.assertIsNone(game.winner(from_text("XO./.X./..O")))

    def test_draw(self):
        b = from_text("XOX/XOO/OXX")
        self.assertIsNone(game.winner(b))
        self.assertEqual(game.status(b), "draw")

    def test_turns(self):
        self.assertEqual(game.next_player(from_text(".........")), X)
        self.assertEqual(game.next_player(from_text("X........")), O)
        self.assertEqual(game.next_player(from_text("XO.......")), X)
        # If O starts, the counts are read the other way round.
        self.assertEqual(game.next_player(from_text("O........"), first_player=O), X)

    def test_impossible_boards(self):
        for text in ("XX.......", "O........", "XXX/OOO/...", "XXX/OO./O.."):
            with self.assertRaises(game.InvalidBoard, msg=text):
                game.check_valid(from_text(text))

    def test_play_does_not_change_original(self):
        b = from_text(".........")
        b2 = game.play(b, 5, X)
        self.assertEqual(b[4], EMPTY)
        self.assertEqual(b2[4], X)
        with self.assertRaises(ValueError):
            game.play(b2, 5, O)

    def test_cell_to_row_col(self):
        self.assertEqual(game.cell_to_row_col(1), (0, 0))
        self.assertEqual(game.cell_to_row_col(6), (1, 2))
        self.assertEqual(game.cell_to_row_col(9), (2, 2))


class TestReferee(unittest.TestCase):
    """Two-player mode: the program checks each move."""

    def test_good_move(self):
        self.assertEqual(game.check_turn(from_text("X........"), from_text("X...O....")), (5, O))

    def test_wrong_turn(self):
        with self.assertRaises(ValueError):
            game.check_turn(from_text("X........"), from_text("X...X...."))

    def test_two_pieces(self):
        with self.assertRaises(ValueError):
            game.check_turn(from_text("........."), from_text("X...O...."))

    def test_nothing_changed(self):
        with self.assertRaises(ValueError):
            game.check_turn(from_text("X........"), from_text("X........"))

    def test_piece_removed_or_changed(self):
        with self.assertRaises(ValueError):
            game.check_turn(from_text("XO......."), from_text("X.O......"))

    def test_game_already_over(self):
        with self.assertRaises(ValueError):
            game.check_turn(from_text("XXX/OO./..."), from_text("XXX/OOO/..."))


class TestComputer(unittest.TestCase):

    def test_takes_the_win(self):
        # O can win on square 6 (middle row).
        self.assertEqual(game.best_move(from_text("XX./OO./X.."), O), 6)

    def test_blocks(self):
        # X threatens square 3; O must block it.
        self.assertEqual(game.best_move(from_text("XX./.O./..."), O), 3)

    def test_prefers_quick_win(self):
        # X can win now on square 3 or 7; it must not play a slower path.
        self.assertEqual(game.best_move(from_text("XX./OO./X.."), X), 3)

    def test_first_move_is_centre(self):
        self.assertEqual(game.best_move(from_text("........."), X), 5)

    def test_answers_corner_with_centre(self):
        self.assertEqual(game.best_move(from_text("X........"), O), 5)

    def test_game_over_has_no_move(self):
        with self.assertRaises(ValueError):
            game.best_move(from_text("XXX/OO./..."))

    def test_perfect_vs_perfect_is_draw(self):
        b = [EMPTY] * 9
        while not game.is_over(b):
            p = game.next_player(b)
            b = game.play(b, game.best_move(b, p), p)
        self.assertEqual(game.status(b), "draw")

    def _all_games_against(self, robot):
        """Let the opponent try EVERY possible sequence of moves; robot must never lose."""
        losses = 0

        def explore(b):
            nonlocal losses
            if game.is_over(b):
                if game.winner(b) == game.other(robot):
                    losses += 1
                return
            p = game.next_player(b)
            if p == robot:
                explore(game.play(b, game.best_move(b, robot), robot))
            else:
                for cell in game.empty_cells(b):
                    explore(game.play(b, cell, p))

        explore([EMPTY] * 9)
        return losses

    def test_never_loses_as_O(self):
        self.assertEqual(self._all_games_against(O), 0)

    def test_never_loses_as_X(self):
        self.assertEqual(self._all_games_against(X), 0)

    def test_easy_sometimes_plays_badly(self):
        b = from_text("XX./.O./...")  # the only good move is 3 (block)
        rng = random.Random(1)
        moves = {game.best_move(b, O, level="easy", easy_random=0.5, rng=rng) for _ in range(50)}
        self.assertIn(3, moves)
        self.assertGreater(len(moves), 1)

    def test_easy_zero_is_hard(self):
        b = from_text("XX./.O./...")
        self.assertEqual(game.best_move(b, O, level="easy", easy_random=0.0), 3)

    def test_random_level_is_legal(self):
        rng = random.Random(0)
        for _ in range(20):
            b = [EMPTY] * 9
            while not game.is_over(b):
                p = game.next_player(b)
                cell = game.best_move(b, p, level="random", rng=rng)
                self.assertIn(cell, game.empty_cells(b))
                b = game.play(b, cell, p)


if __name__ == "__main__":
    unittest.main()
