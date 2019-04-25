from math import *
import pandas as pd
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import random
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is player 2 - player 1 has the first move

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class NimState:
    """ A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips with the
        winner being the player to take the last chip.
        In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for player 1
        (by choosing k) chips.
        Any initial state of the form 4n is a win for player 2.
    """

    def __init__(self, ch):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.chips = ch

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.chips)
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        assert move >= 1 and move <= 3 and move == int(move)
        self.chips -= move
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return range(1, min([4, self.chips + 1]))

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        assert self.chips == 0
        if self.playerJustMoved == playerjm:
            return 1.0  # playerjm took the last chip and has won
        else:
            return 0.0  # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Chips:" + str(self.chips) + " JustPlayed:" + str(self.playerJustMoved)
        return s


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5  # draw
        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class OthelloState:
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move).
    """

    def __init__(self, sz=8):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for y in range(sz):
            self.board.append([0] * sz)
        self.board[sz / 2][sz / 2] = self.board[sz / 2 - 1][sz / 2 - 1] = 1
        self.board[sz / 2][sz / 2 - 1] = self.board[sz / 2 - 1][sz / 2] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(x, y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x, y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a, b) in m:
            self.board[a][b] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size) if
                self.board[x][y] == 0 and self.ExistsSandwichedCounter(x, y)]

    def AdjacentToEnemy(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x + dx, y + dy) and self.board[x + dx][y + dy] == self.playerJustMoved:
                return True
        return False

    def AdjacentEnemyDirections(self, x, y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.IsOnBoard(x + dx, y + dy) and self.board[x + dx][y + dy] == self.playerJustMoved:
                es.append((dx, dy))
        return es

    def ExistsSandwichedCounter(self, x, y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            if len(self.SandwichedCounters(x, y, dx, dy)) > 0:
                return True
        return False

    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.AdjacentEnemyDirections(x, y):
            sandwiched.extend(self.SandwichedCounters(x, y, dx, dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x, y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.IsOnBoard(x, y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        jmcount = len([(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len(
            [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size - 1, -1, -1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        pass
        #return ""#[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
        #    self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s

def UCT(dectree,rootstate, itermax,verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0].
        """

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            randdbl = random.uniform(0,1)
            if randdbl<0.9:
                playerAndBoard = []
                playerAndBoard.append(state.playerJustMoved)
                playerAndBoard.extend(state.board)
                statespace =[]
                statespace.append(playerAndBoard)
                prediction = dectree.predict(statespace)
                if(playerAndBoard[prediction[0]+1]==0):
                    state.DoMove(prediction[0])
                else:
                    # if not predicted correctly then choose random
                    state.DoMove(random.choice(state.GetMoves()))
            else:
                state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(
                node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose):
        pass
        #print(rootnode.TreeToString(0))
        print(rootnode.ChildrenToString())


    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move  # return the move that was most visited

def logState(state, move,data_frame):

    player = None
    if state.playerJustMoved==2:
        player = 1
    else:
        player = 2
    player_state_move_endgame = {'player':player}
    player_state_move_endgame['0'] = state.board[0]
    player_state_move_endgame['1'] = state.board[1]
    player_state_move_endgame['2'] = state.board[2]
    player_state_move_endgame['3'] = state.board[3]
    player_state_move_endgame['4'] = state.board[4]
    player_state_move_endgame['5'] = state.board[5]
    player_state_move_endgame['6'] = state.board[6]
    player_state_move_endgame['7'] = state.board[7]
    player_state_move_endgame['8'] = state.board[8]
    player_state_move_endgame['best_move'] = str(move)
    df = data_frame.append(player_state_move_endgame,sort=False,ignore_index=True)
    return df

def updateWinField(data_frame,won):
    data_frame['won'] = won
    return data_frame

def concat_data_frames(df,data_frame):
    frames = [df,data_frame]
    return pd.concat(frames)

def UCTPlayGame(data_frame,dectree,windrawloss):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState() # uncomment to play OXO
    df = pd.DataFrame()
    while (state.GetMoves() != []):
        m = None
        if state.playerJustMoved == 1:
            m = UCT(dectree, rootstate=state, itermax=1000, verbose=False,)  # play with values for itermax and verbose = True
        else:
            m = UCT(dectree, rootstate=state, itermax=100, verbose=False)
        df = logState(state, m, df)
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
        df = updateWinField(df,state.playerJustMoved)
        windrawloss.append(1)
        return concat_data_frames(df,data_frame)
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
        df = updateWinField(df,str(3-state.playerJustMoved))
        windrawloss.append(2)
        return concat_data_frames(df,data_frame)
    else:
        print("Nobody wins!")
        df = updateWinField(df,0)
        windrawloss.append(0)
        return concat_data_frames(df,data_frame)

def visualise_tree(trained_tree):
    dot_data = tree.export_graphviz(trained_tree,out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("oxo")

def trainTree(training_values,prediction_values):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(training_values,prediction_values)
    return clf

def writeCsv(name,df):
    df.to_csv(name)

def getSlicesOfData(read_csv):
    slice_training_data = read_csv[["player", "0", "1", "2", "3", "4", "5", "6", "7", "8"]]
    slice_prediction_data = read_csv[["best_move"]]
    return (slice_training_data, slice_prediction_data)

def getDataSplit(data_sliced):
    X_train, X_test, Y_train, Y_test = train_test_split(data_sliced[0],data_sliced[1], test_size=0.3,random_state=0)
    return {'x_train' : X_train,'x_test' : X_test,'y_train' : Y_train, 'y_test':Y_test}

def play_a_hundred_games(dectree):
    print("playing a hundred games with optimised tree")
    df = pd.DataFrame()
    windrawloss = []
    for i in range(0, 100):
        UCTPlayGame(df, dectree,windrawloss)
    count_number_of_wins_and_draws(windrawloss)

def count_number_of_wins_and_draws(win_draw_loss):
    player1_win = win_draw_loss.count(1)
    player2_win = win_draw_loss.count(2)
    draw = win_draw_loss.count(0)
    print("The number of times player 1 won was " + str(player1_win) + " and the number of times player 2 won was " + str(player2_win) + " and the number of times a draw occurred was " + str(draw))

def run_n_iterations(number_of_games,number_of_epochs,training_slice,prediction_slice):
    initial_X_train, post_X_test, initial_y_train, post_y_test = train_test_split(training_slice, prediction_slice, test_size=0.3, random_state=0)
    dectree = trainTree(initial_X_train,initial_y_train)
    print("initial score on test data " + str(dectree.score(post_X_test,post_y_test)))
    epoch_trees = []
    epoch_accuracy = []
    epoch_scores = []
    predictions = []
    x_test_array = []
    print("initial game")
    play_a_hundred_games(dectree)


    # over a number of epochs
    for epoch in range(0,number_of_epochs):
        data_frame = pd.DataFrame()
        win_draw_loss = []
        print("start of epoch " + str(epoch) + ":")
        #play n games, collecting values for game outcomes with a dummy data frame.
        for i in range(0,number_of_games):
            data_frame = UCTPlayGame(data_frame,dectree,win_draw_loss)
        count_number_of_wins_and_draws(win_draw_loss)
        writeCsv("epoch.csv", data_frame)
        read_csv = pd.read_csv("epoch.csv")
        #read_csv = filter_winning_combinations(read_csv)
        #read_csv = read_csv.drop_duplicates()
        training_slice, prediction_slice = getSlicesOfData(read_csv)
        X_train, X_test, y_train, y_test = train_test_split(training_slice, prediction_slice, test_size=0.3, random_state=0)
        dectree = trainTree(X_train, y_train)
        predictions.append(dectree.predict(X_test))
        x_test_array.append(y_test)
        score = dectree.score(X_test, y_test)
        print("score on epoch " + str(epoch) + " " + str(score))
        epoch_trees.append(dectree)
        epoch_accuracy.append(score)
        epoch_scores.append(win_draw_loss.count(0))

    optimised_tree = epoch_trees[epoch_scores.index(min(epoch_scores))]
    print(epoch_scores)
    print("index of minimum draws " + str(epoch_scores.index(min(epoch_scores))))
    print("testing optimised tree scored from original test data. The MaxScore is: " + str(max(epoch_accuracy)) +  ";score on original data is: " + str(optimised_tree.score(post_X_test,post_y_test)))
    play_a_hundred_games(optimised_tree)
    index_of_max_accuracy = epoch_accuracy.index(max(epoch_accuracy))
    print("index of best accuracy score " + str(index_of_max_accuracy))
    print("playing a hundred games with tree that achieved max accuracy")
    play_a_hundred_games(epoch_trees[epoch_accuracy.index(max(epoch_accuracy))])
    print("play last tree in list")
    play_a_hundred_games(epoch_trees[-1])


#focus on winning combinations
def filter_winning_combinations(csv_df):
    csv_df = csv_df[csv_df.won!=0]
    return csv_df

if __name__ == "__main__":
    read_csv = pd.read_csv('10000games.csv')
    #read_csv = filter_winning_combinations(read_csv)
    #read_csv = read_csv.drop_duplicates()
    training_slice,prediction_slice = getSlicesOfData(read_csv) #limit to winning games?
    run_n_iterations(500,10,training_slice,prediction_slice)