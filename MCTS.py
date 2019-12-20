# Adapted from original code by Christopher Yong, who modified it from mcts.ai (defunct website)

import copy
import math
import random
import time
import sys

import pickle
from sklearn import preprocessing
import numpy as np

# Dictionary for returning proper value to backpropagated when predicting the outcome
predictions = {
  "win": [1, 0],
  "loss": [0, 1],
  "draw": [0.5, 0.5]
}

r_forest_sim = 250      # How many simulations to run with random forest
human_player = 1        # Set to 1 to play against the AI (0 will pit two AI against each other)
order = 0               # Who goes first? (set to 0 or 1)
use_random_forest = 1   # Set to 0 or 1 (determines whether random forest predictions are used, or pure monte carlo)

# ==========================================================================

print("Fitting data...")
start = time.time()

# Only this line to load the model from file
loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
# Define the possible categories for each feature
feature_categories = [['b', 'o', 'x'] for i in range(42)]
# Encoder for encoding the prediction string
enc = preprocessing.OneHotEncoder(categories=feature_categories)

print("Completed in", time.time() - start, "seconds.")

# ==========================================================================

def update_predictions(pred, player):

    global predictions

    win = pred[0][2]
    loss = pred[0][1]
    draw = pred[0][0]
    outcome = ["draw", "loss", "win"]

    predictions = {
        "win": [win, 0],
        "loss": [0, loss],
        "draw": [draw, draw]
    }

    return outcome[np.argmax(pred)]


# ==========================================================================


def monte_carlo(current_state, max_iterations, current_node=None, timeout=100.0):

    global rng

    root = Node(state=current_state)

    if current_node is not None:
        root = current_node

    start = time.time()
    pred = None

    for i in range(max_iterations):
        node = root
        state = current_state.clone()

        # Selection --
        # Keep traversing the tree based on the best UCT values until
        # reaching a terminal state or an unexpanded node

        while node.untried_moves == [] and node.child_nodes != []:
            node = node.selection()
            state.move(node.move)

        # Expansion
        if node.untried_moves:
            m = random.choice(node.untried_moves)
            state.move(m)
            node = node.expand(m, state)

        # Simulation
        while state.get_moves():

            state.move(random.choice(state.get_moves()))

            if state.depth == 6 and use_random_forest:

                if state.winner(1, lines=3):
                    print("Red has three in a row after 8-ply in a simulated game!  Cannot predict the result!\n")
                else:

                    # Have to predict this board using other algorithm based on the Connect4 dataset
                    # then we need to insert the most likely result into   state.result(node.player)
                    # and break off this loop, then it should work as is for the backpropagation loop ...

                    prediction_string = np.asanyarray(state.prediction_string())    # Load the board positions and convert
                                                                                    # into into a numpy array

                    # Reshape the array into a one sample array and encode it
                    pred_str_transformed = enc.fit_transform(prediction_string.reshape(1, -1))

                    # Predict outcome
                    pred = loaded_model.predict_proba(pred_str_transformed)
                    pred = update_predictions(pred, node.player)

        # Backpropagation
        while node is not None:
            node.update(state.result(node.player, pred))
            node = node.parent

        duration = time.time() - start

        if duration > timeout:
            if i >= r_forest_sim:  # Let the AI try a minimum of simulations during the random forest phase
                break

    foo = lambda x: x.wins / x.simulations
    sorted_child_nodes = sorted(root.child_nodes, key=foo)[::-1]

    print("AI's computed winning percentages:")

    for node in sorted_child_nodes:
        print("Move: %s    Win Rate: %.2f%%" % (node.move + 1, 100 * node.wins / node.simulations))

    print("Simulations performed: %s\n" % i)

    return root, sorted_child_nodes[0].move


# ==========================================================================


class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.state = state.clone()
        self.parent = parent
        self.move = move
        self.untried_moves = state.get_moves()
        self.child_nodes = []
        self.wins = 0
        self.simulations = 0
        self.player = state.player

    def selection(self):
        # Return the child with the highest UCT value
        foo = lambda x: x.wins / x.simulations + math.sqrt(2 * math.log(self.simulations) / x.simulations)
        return sorted(self.child_nodes, key=foo)[-1]

    def expand(self, move, state):
        # Return child node when expanding the tree, then remove this move from the parent node
        child = Node(move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.child_nodes.append(child)

        return child

    def update(self, result):
        self.wins += result
        self.simulations += 1


# ==========================================================================


class Board:
    def __init__(self, depth=0, row=6, column=7, line=4):
        self.bitboard = [0, 0]
        self.dirs = [1, (row + 1), (row + 1) - 1, (row + 1) + 1]
        self.heights = [(row + 1) * i for i in range(column)]
        self.lowest_row = [0] * column
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        self.top_row = [(x * (row + 1)) - 1 for x in range(1, column + 1)]
        self.row = row
        self.column = column
        self.line = line
        self.player = 1
        self.depth = depth                              # To enable us to use the prediction when we reach depth=6

    def prediction_string(self):
        tromp = []

        for column in range(self.column):
            for row in range(self.row):
                output = 'bxo'[self.board[row][column]]
                tromp.append(output)

        return tromp

    def print_variables(self):
        print("bitboard: ", self.bitboard)
        print("dirs: ", self.dirs)
        print("heights: ", self.heights)
        print("lowest_row: ", self.lowest_row)
        print("board: ", self.board)
        print("top_row: ", self.top_row)
        print("row: ", self.row)
        print("column: ", self.column)
        print("line: ", self.line)
        print("player: ", self.player)
        print("depth: ", self.depth)

    def clone(self):
        clone = Board(self.depth, self.row, self.column, self.line)
        clone.bitboard = copy.deepcopy(self.bitboard)
        clone.heights = copy.deepcopy(self.heights)
        clone.lowest_row = copy.deepcopy(self.lowest_row)
        clone.board = copy.deepcopy(self.board)
        clone.top_row = copy.deepcopy(self.top_row)
        clone.player = self.player
        clone.depth = self.depth   # self.depth + 1   ?
        return clone

    def move(self, column):
        m2 = 1 << self.heights[column]      # Position on bitboard
        self.heights[column] += 1           # Update top empty row for this column
        self.player ^= 1
        self.bitboard[self.player] ^= m2    # XOR operation to insert disc in this player's bitboard
        self.board[self.lowest_row[column]][column] = self.player + 1   # Update matrix
        self.lowest_row[column] += 1        # Update the number of discs in this column
        self.depth += 1

    def result(self, player, prediction=None):

        if prediction is not None:
            return predictions.get(prediction)[player]

        if self.winner(player):         # Player wins
            return 1
        elif self.winner(player ^ 1):   # The opponent wins
            return 0
        elif self.draw():               # The game resulted in a draw
            return 0.5

    def is_valid_move(self, column):  # Check if this column is full
        return self.heights[column] != self.top_row[column]

    def winner(self, color, lines=4):  # Evaluate board to check if there is a winner
        for d in self.dirs:
            bb = self.bitboard[color]

            # for i in range(1, self.line):
            for i in range(1, lines):
                bb &= self.bitboard[color] >> (i * d)

            if bb != 0:
                return True

        return False

    def print_board(self):

        board = copy.deepcopy(self.board)
        board.reverse()

        for row in board:
            sys.stdout.write("\t")

            for column in row:
                output = '.YR'[column]
                sys.stdout.write(output + " ")

            sys.stdout.write("\n")

        sys.stdout.write("\t")

        for i in range(1, self.column + 1):
            sys.stdout.write(str(i) + " ")

        sys.stdout.write("\n\n")

    def draw(self):  # Check if this is a draw
        return not self.get_moves() and not self.winner(self.player) and not self.winner(self.player ^ 1)

    def complete(self):  # Check if the game has ended
        return self.winner(self.player) or self.winner(self.player ^ 1) or not self.get_moves()

    def get_moves(self):

        # Return an empty list for a terminal state
        if self.winner(self.player) or self.winner(self.player ^ 1):
            return []

        list_moves = []

        for i in range(self.column):
            if self.lowest_row[i] < self.row:
                list_moves.append(i)

        return list_moves


# ==========================================================================

def get_input(board):
    while True:
        try:
            cin = int(input("Your move, [1-%s]. Ans: " % board.column))

            if cin == -1:  # Entering '-1' exits the game
                sys.exit()
            if cin < 1 or cin > board.column:
                raise ValueError
            if not board.is_valid_move(cin - 1):
                print(cin)
                raise ValueError
            print()
            return cin - 1
        except ValueError:
            print("Invalid move. Try again.")

def play_game(board, order=0, max_iterations=20000, timeout=2.0):
    players = ["Player 1", "Player 2"]
    time_allocation = [2.0, 2.0]

    node = Node(state=board)

    print(board.board)

    while True:
        if order == 0:

            if human_player:
                column = get_input(board)
            else:
                print("Player 1 is thinking...")
                node, column = monte_carlo(board, max_iterations, current_node=node, timeout=time_allocation[0])
                print("Player 1 played column %s\n" % (column + 1))

        elif order == 1:

            print("Player 2 is thinking...")
            node, column = monte_carlo(board, max_iterations, current_node=node, timeout=time_allocation[1])
            print("Player 2 played column %s\n" % (column + 1))


        board.move(column)
        board.print_board()
        print(board.board)
        node = goto_child_node(node, board, column)

        order ^= 1

        if board.complete():
            break

    if not board.draw():
        print("%s won!" % players[board.player])
    else:
        print("Draw")


def goto_child_node(node, board, move):
    for child_node in node.child_nodes:
        if child_node.move == move:
            return child_node

    return Node(state=board)


print("\nConnect Four\n")
Connect4 = Board()  # Create board object
Connect4.print_board()

max_iterations = 100000  # 10000
timeout = 3
play_game(Connect4, order, max_iterations, timeout)  # Play a game
