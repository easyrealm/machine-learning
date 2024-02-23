import numpy as np
import random
from collections import defaultdict

DIMENSION = 3

class TicTacToe:
    def __init__(self) -> None:
        self.dimension = DIMENSION
        self.board = np.zeros((self.dimension, self.dimension), dtype=int)
        self.game_over = False
        self.winner = None
        self.current_player = 1 #  2 players -> 1, -1

    def reset(self) -> None:
        self.board.fill(0)
        self.game_over = False
        self.winner = None
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        state = tuple(self.board.reshape(self.dimension * self.dimension))
        return state

    def step(self, action):
        row, col = action // self.dimension,  action % self.dimension
        if self.board[row, col] != 0 or self.game_over:
            raise ValueError('Invalid move')
        
        self.board[row, col] = self.current_player
        self.current_player *= -1
        self.check_game_over()

        reward = 0
        if self.game_over:
            if self.winner is None:
                reward = 0.5
            else:
                reward = self.winner

        new_state = tuple(self.board.copy().reshape(self.dimension * self.dimension))
        return new_state, reward, self.game_over
    
    def check_game_over(self):
        for i in range(3):
            if (abs(sum(self.board[i, :]))) == self.dimension or (abs(sum(self.board[:, i]))) == self.dimension:
                self.game_over = True
                self.winner = np.sign(sum(self.board[i, :])) or np.sign(sum(self.board[:, i]))
                return

        if (abs(sum(np.diag(self.board)))) == self.dimension or (abs(sum(np.diag(np.fliplr(self.board))))) == self.dimension:
            self.game_over = True
            self.winner =  np.sign(sum(np.diag(self.board))) or  np.sign(sum(np.diag(np.fliplr(self.board))))
            return
        
        if 0 not in self.board:
            self.game_over = True
            self.winner = None

    def is_winner(self, player):
        return self.winner is not None and self.winner == player
    
    def is_draw(self):
        return self.winner is None

    def get_available_actions(self):
        return [i for i in range(self.dimension * self.dimension) if self.board[i // self.dimension, i % self.dimension] == 0]
    

class  QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1) -> None:
        self.q = defaultdict(float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            # find the max Q value
            q_values = {action: self.get_q(state, action) for action in available_actions}
            max_q = max(q_values.values())
            
            # find the right action that has the max Q
            max_actions = [action for action, q_value in q_values.items() if q_value == max_q]
            return random.choice(max_actions)
        
    def update_q(self, state, action, reward, next_state, done):
        curren_q = self.get_q(state, action)
        max_q_next = max([self.get_q(next_state, a) for a in range(DIMENSION * DIMENSION)]) if not done else 0.0
        self.q[(state, action)] += curren_q + self.learning_rate * (reward + self.discount_factor * max_q_next - curren_q)
        