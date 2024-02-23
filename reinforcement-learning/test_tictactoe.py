import unittest
import numpy as np
from tictactoe import TicTacToe

class TestTicTacToe(unittest.TestCase):

    def test_board_initialization(self):
        game = TicTacToe()
        expected_board = np.zeros((3, 3), dtype=int)
        np.testing.assert_array_equal(game.board, expected_board)

    def test_game_reset(self):
        game = TicTacToe()
        game.board.fill(1)
        game.reset()
        expected_board = np.zeros((3, 3), dtype=int)
        np.testing.assert_array_equal(game.board, expected_board)
        self.assertFalse(game.game_over)
        self.assertIsNone(game.winner)
        self.assertEqual(game.current_player, 1)

    def test_get_state(self):
        game = TicTacToe()
        state = game.get_state()
        expected_state = tuple(np.zeros(9, dtype=int))
        self.assertEqual(state, expected_state)

    def test_step_valid_move(self):
        game = TicTacToe()
        action = 0  # Top-left corner
        state, reward, game_over = game.step(action)
        self.assertEqual(state[0], 1)
        self.assertEqual(reward, 0)
        self.assertFalse(game_over)

    def test_step_invalid_move(self):
        game = TicTacToe()
        action = 0  # Top-left corner
        game.step(action)  # First move
        with self.assertRaises(ValueError):
            game.step(action)  # Invalid move, cell already taken

    def test_win_condition(self):
        game = TicTacToe()
        game.board = np.array([[1,1,1],
                               [0,-1,0],
                               [0,0,-1]])
        game.check_game_over()
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

    def test_draw_condition(self):
        game = TicTacToe()
        game.board = np.array([[1,-1,1],
                               [1,1,-1],
                               [-1,1,-1]])
        game.check_game_over()
        self.assertTrue(game.game_over)
        self.assertIsNone(game.winner)

    def test_is_winner(self):
        game = TicTacToe()
        game.board = np.array([[1, 1, 1],
                               [0, -1, -1],
                               [0, 0, 0]])
        game.check_game_over()
        self.assertTrue(game.is_winner(1))
        self.assertFalse(game.is_winner(-1))

    def test_get_available_actions(self):
        # Set a partially complete board and check available actions
        game = TicTacToe()
        game.board = np.array([[1, 0, -1],
                               [0, -1, 0],
                               [1, 0, 0]])
        expected_actions = [1, 3, 5, 7, 8]
        self.assertEqual(game.get_available_actions(), expected_actions)


if __name__ == '__main__':
    unittest.main()
