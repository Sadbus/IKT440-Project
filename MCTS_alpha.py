import copy
import math
import random
import time

# This is our own, buggy implementation of the MCTS algorithm for playing Connect Four
# Due to time constraints, we abandoned it and used Christopher Yung's version instead.

c = math.sqrt(2)
debug = 0

# board, represented as  board[y][x], where y element [0, 5] and x element [0, 6]
board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

# ==========================================================================

class MonteCarlo:

    def __init__(self, _board):
        empty_board = Board(_board, 0)
        self.current_node = Node(0, empty_board, empty_board)
        pass

    def play(self):

        # Should the whole thing start from scratch for every move?
        # If so, then all the old nodes should be deleted for memory reasons (?)

        play_time = [1, 0.2]
        iterations = 0

        print(self.current_node.board.print_board())

        player = self.current_node.depth % 2

        print("Player: ", player)

        t_start = time.time()

        while time.time() - t_start < play_time[player]:
            self.selection(self.current_node)
            iterations += 1

        print("Finished playing ", iterations, "simulations in ", time.time() - t_start, " seconds.")

        # Now print out the UCT scores for all the child nodes of the current position ...

        best_move = []

        for i in range(len(self.current_node.children)):
            child_node = self.current_node.children[i]

            print("Node: ", i, "  Column: ", child_node.move, "  Simulations: ", child_node.simulations, "  Wins: ",
                  child_node.wins, "  UCT score: ", self.calculate_uct(child_node))

            best_move.append(child_node.simulations)

        index = best_move.index(max(best_move))
        print("The best choice is to play Column: ", self.current_node.children[index].move, "(Node: ", index, ")")

        self.current_node = self.current_node.children[index]

        '''
        self.current_node.children = []   # Let us try this...
        self.current_node.simulations = 0
        self.current_node.wins = 0
        '''

        # Debug
        print(self.current_node.board.board)

        # Is the game won, or has the board been filled?

        if self.current_node.board.won():
            print("Player ", player, " wins the game!")
            print(self.current_node.board.print_board())
            quit()

        self.play()

    def calculate_uct(self, node):
        if node.simulations:  # self.simulations has to be greater than 0
            uct = (node.wins / node.simulations) + c * math.sqrt(math.log(node.parent.simulations) / node.simulations)
        else:
            uct = math.inf  # If this has not been simulated yet, we set the UCT score to infinity

        return uct

    def selection(self, node):

        # Starting from root node, go down the tree by repeatedly
        # 1) selecting a legal move
        # 2) advancing to corresponding child node

        # if one, several, or all of the legal moves in a node does not have a corresponding node in the search tree,
        # we stop selection

        # Check if this node has child nodes for all the possible legal moves ...

        if node.exhausted():

            # Use the UCT algorithm to select the best choice here ... then recursively feed that choice into
            # selection

            selection = [-math.inf] * len(node.children)

            for i in range(len(node.children)):

                child_node = node.children[i]

                # Let us see if this helps...  This only checks one possibility ?
                # This will also pull the ... fix, maybe just if it is the following turn from current_node's

                # if node == self.current_node or self.current_node == node.parent:

                '''
                selection[i] = self.calculate_uct(child_node)

                if node == self.current_node:
                    if child_node.board.won() or child_node.adv_board.won():
                        selection[i] = math.inf
                elif self.current_node == node.parent:
                    if child_node.board.won() or child_node.adv_board.won():
                        selection[i] = -math.inf
                
                '''


                if node == self.current_node:
                    if child_node.board.won() or child_node.adv_board.won():
                        child_node.wins = math.inf
                elif self.current_node == node.parent:
                    if child_node.board.won() or child_node.adv_board.won():

                        # This can still be influenced by later play, so it could get the most simulations

                        # child_node.wins = -math.inf  # -math.inf
                        child_node.wins = 0  # -math.inf
                        child_node.simulations = 1

                        return


                selection[i] = self.calculate_uct(child_node)


                if debug:
                    print(f"Child node {i} has UCT score of {selection[i]}")

            if debug:
                print("selection: ", selection)

            if selection:
                selected = selection.index(max(selection))

                if debug:
                    print("Child node ", selected, " wins the selection...")

                self.selection(node.children[selected])  # Recursively input the node that wins the selection
            else:

                '''
                print("\nSELECTION IS EMPTY!!!\n")
                print("Length of children: ", len(node.children))
                print("Legal moves of this board: ", node.board.legal_moves())
                print("Node board: ", node.board.board)
                print("Depth of node: ", node.depth)
                '''
                # This happens when you get to a draw state where the entire board has been filled up
                # We can not keep going, and cannot expand since there are no more available nodes to expand.

                # We could possibly just start the simulation?

                self.simulation(node)
        else:
            self.expansion(node)

    def expansion(self, node):

        # There are unexplored nodes here ... Pick one at random from the legal moves ...
        #  Create that node, and feed it into the expansion method

        # Have to know for sure if this node does not exist yet...

        # exists = []
        # for i in range(node.board.legal_moves()):

        legal_columns = node.board.legal_moves()

        if debug:
            print("Non-exhausted -- legal_moves: ", legal_columns)

        # debug
        # new_node = Node(node.depth + 1, node.board, node, 1)
        # node.children.append(new_node)

        for i in range(len(node.children)):
            if node.children[i].move in legal_columns:

                if debug:
                    print("a child node already has column", node.children[i].move, "  removing it from legal_columns")
                legal_columns.remove(node.children[i].move)

        if debug:
            print("legal_columns after pruning: ", legal_columns)

        r = random.randint(0, len(legal_columns) - 1)

        # We now have a random node we can expand ... Create it
        # def get_next_board(self, position):  # Input a tuple here in the form (y, x)

        if debug:
            print(legal_columns[r])

        new_board = node.board.get_next_board(legal_columns[r])
        adv_board = node.board.get_next_board(legal_columns[r], -1)

        if debug:
            print(new_board.print_board())

        # Create a new node
        new_node = Node(node.depth + 1, new_board, adv_board, node, legal_columns[r])
        # Update the parent with this as a child node
        node.children.append(new_node)

        # If this node ends the game decidedly (win/loss) it should be played ...

        self.simulation(new_node)  # Run a simulation with this new node


    def simulation(self, node):

        # Continuing from the newly-created node in the expansion phase, moves are selected randomly and the game
        # state is repeatedly advanced.  This repeates until the game is finished and a winner emerges

        # No new nodes are created in this phase

        # Take the board as it is here ... play random moves and get new boards until the board is either full (draw)
        # or there is a winner ...

        _sim_board = copy.deepcopy(node.board)

        winner = False
        won_by = -1

        # while _sim_board.depth <= 42 and winner is False:
        while winner is False:

            if debug:
                print("Board:")
                print(_sim_board.print_board())

            # Get the legal plays from this board

            legal = _sim_board.legal_moves()

            if debug:
                print("len(legal): ", len(legal))

            if len(legal) == 0:  # We have filled the board, there is no winner
                break

            r = random.randint(0, len(legal) - 1)

            if debug:
                print("legal_moves: ", legal)

            # play a random move, and get a new board
            _sim_board = _sim_board.get_next_board(legal[r])

            winner = _sim_board.won()

        if debug:
            print("Player 0 is Yellow, Player 1 is Red")
            print("Final Board Position:  (Winner: ", winner, "), depth: ", _sim_board.depth)
            print("Who won? ", (_sim_board.depth + 1) % 2)
            # print("Who won? ", _sim_board.depth % 2)
            print(_sim_board.print_board())

        if winner:
            won_by = (_sim_board.depth + 1) % 2

        # Send the winner into the backpropagation method

        self.backpropagation(node, won_by)


    def backpropagation(self, node, winner):

        while True:

            if debug:
                print("@@ Winner: ", winner)

            node.simulations += 1

            if debug:
                print("This node was played by ", (node.depth + 1) % 2)

            # This may be reversed, check later...
            # if (node.depth + 1) % 2 == 1:
            #     node.wins += 1

            wins_before = node.wins

            if winner >= 0:  # Do not add any wins for a draw
                if (node.depth + winner) % 2 == 1:
                    node.wins += 1

            if debug:
                print("Node.depth: ", node.depth, " node.simulations: ", node.simulations,
                      "  node.wins (before/after): ",
                      wins_before, "/", node.wins)

            if node.parent is None:
                break

            node = node.parent

        if debug:
            print("Backpropagation complete.")



# ==========================================================================

class Node:
    def __init__(self, depth, _board, adv_board, parent=None, move=None):
        self.parent = parent
        self.depth = depth
        self.board = _board
        self.adv_board = adv_board
        self.move = move    # Which column was used last?

        self.children = []
        self.wins = 0
        self.simulations = 0

        if debug:
            print("Init node.")

    def exhausted(self):  # There are children nodes for all the legal moves in this node

        if len(self.children) == len(self.board.legal_moves()):
            return True

        return False

# ==========================================================================

class Board:
    # What should be in the board?
    # Functions to get possible columns for drops
    # Function to return the new board with the move in place
    # Function to check if a board has been won / lost / draw

    def __init__(self, o_board, depth):
        self.board = o_board
        self.depth = depth

    def get_player_score(self):  # Gets 1 for player 0, and -1 for player 1
        if self.depth % 2 == 1:
            return -1
        return 1

    def get_row(self, column):
        row = 5
        while self.board[row][column] != 0:
            row -= 1
        return row


    def legal_moves(self):
        legal_moves = []

        for x in range(7):
            if self.board[0][x] == 0:
                #legal_moves.append((self.get_row(x), x))  # We are adding a tuple to the list in the form (y, x)
                legal_moves.append(x)  # We are adding an x value to the list

        return legal_moves  # Returning a list of tuples

    def get_next_board(self, column, adversary=1):  # Input an x value here for column
        next_b = copy.deepcopy(self.board)
        next_b[self.get_row(column)][column] = self.get_player_score() * adversary   # (y, x)
        next_board = Board(next_b, self.depth + 1)
        return next_board  # Returning a new board object with the ply played

    def won(self):
        # Check to see if this board is a winning position

        for y in range(6):
            for x in range(7):

                # Check Horizontal
                if x <= 3:
                    if abs(self.board[y][x] + self.board[y][x + 1] +
                           self.board[y][x + 2] + self.board[y][x + 3]) == 4:
                        return True

                # Check Vertical
                if y <= 2:
                    if abs(self.board[y][x] + self.board[y + 1][x] +
                           self.board[y + 2][x] + self.board[y + 3][x]) == 4:
                        return True

                # Check SE diagonal
                if y <= 2 and x <= 3:
                    if abs(self.board[y][x] + self.board[y + 1][x + 1] +
                           self.board[y + 2][x + 2] + self.board[y + 3][x + 3]) == 4:
                        return True

                # Check NE diagonal
                if y >= 3 and x <= 3:
                    if abs(self.board[y][x] + self.board[y - 1][x + 1] +
                           self.board[y - 2][x + 2] + self.board[y - 3][x + 3]) == 4:
                        return True

        return False

    def print_board(self):

        b_string = ""

        for y in range(6):
            for x in range(7):
                if self.board[y][x] == 1:
                    b_string += "Y "
                elif self.board[y][x] == -1:
                    b_string += "R "
                else:
                    b_string += ". "

            b_string += "\n"

        return b_string + "0 1 2 3 4 5 6"


MCTS = MonteCarlo(board)
MCTS.play()
