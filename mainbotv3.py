

import timeit
from board import *
from numba.core import types
from numba.typed import Dict
from numba.experimental import jitclass
from numba import int32, float64

tt_table_type = (types.unicode_type, types.int64[:])
wt_table_type = (types.unicode_type, types.int64)
spec = [
    ('TRANSPOSITION_TABLE', types.DictType(*tt_table_type)),
    ('WIN_TABLE', types.DictType(*wt_table_type)),
    ('max_depth', int32),
    ('min_stop_depth', int32),
    ('curr_depth', int32),
    ('start_time', float64),
    ('node_count', int32)
]


class MainAiV3:

    def __init__(self):
        self.TRANSPOSITION_TABLE = {}
        self.WIN_TABLE = {}

        self.max_depth = 15
        self.min_stop_depth = 1
        self.curr_depth = 0

        self.start_time = 0
        self.node_count = 0

    def iterative_deepening(self, board, plr, last_move):

        self.start_time = timeit.default_timer()

        np_board = numpy.array(board)
        node_sum = 0
        best_return = []
        for running_depth in range(1, self.max_depth + 1):
            self.node_count = 0
            self.curr_depth = running_depth

            returned = self.negamax(np_board, plr, MIN, MAX, last_move[0], last_move[1], self.curr_depth+1)

            node_sum += self.node_count
            if returned[0] != [-2, -2]:
                best_return = [returned[0], returned[1] if plr == 1 else -returned[1]]
                print(self.curr_depth, timeit.default_timer() - self.start_time,
                      best_return[0], best_return[1]/100, self.node_count, node_sum)
            else:
                print(self.curr_depth, "UNFINISHED", self.node_count, node_sum)
                return best_return
        return best_return

    def negamax(self, board, plr, alpha, beta, last_y, last_x, depth):

        tt_code = np_tt_hash(board)
        if tt_code in self.WIN_TABLE:
            test_result = self.WIN_TABLE[tt_code]
        else:
            test_result = np_test_for_win_enhanced(board, WINNEED, last_y, last_x)
            self.WIN_TABLE[tt_code] = test_result
        if test_result:
            return [-1, -1], -10000 - depth

        if depth == 0:
            return [-1, -1], np_evaluate_neg(board, plr)

        moves = np_find_ordered_moves(board)
        if len(moves) == 0:
            return [-1, -1], 0
        if timeit.default_timer() - self.start_time >= MAXTIME and self.curr_depth >= self.min_stop_depth:
            return [-2, -2], 0

        alpha_org = alpha
        pv_move = []

        if tt_code in self.TRANSPOSITION_TABLE:
            tt_entry = self.TRANSPOSITION_TABLE[tt_code]
            if tt_entry[4] >= depth:
                if tt_entry[3] == 0:  # checking for Exact
                    return [tt_entry[0], tt_entry[1]], tt_entry[2]
                elif tt_entry[3] == -1:  # lower bound
                    alpha = max(alpha, tt_entry[2])
                elif tt_entry[3] == 1:  # upper bound
                    beta = min(beta, tt_entry[2])

                if alpha >= beta:
                    return [tt_entry[0], tt_entry[1]], tt_entry[2]
            else:
                pv_move = [tt_entry[0], tt_entry[1]]

        if pv_move:
            moves.insert(0, moves.pop(moves.index(pv_move)))

        self.node_count += 1

        best_move = []
        best_score = -100000

        for move in moves:
            board[move[0], move[1]] = plr
            returned = self.negamax(board, 2 if plr == 1 else 1, -beta, -alpha, move[0], move[1], depth-1)
            board[move[0], move[1]] = 0

            if returned[0] == [-2, -2]:
                return [-2, -2], 0

            return_eval = -returned[1]
            if return_eval > best_score:
                best_score = return_eval
                best_move = move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        if best_score <= alpha_org:
            flag = 1  # Upper bound
        elif best_score >= beta:
            flag = -1  # Lower bound
        else:
            flag = 0  # Exact

        self.TRANSPOSITION_TABLE[tt_code] = [best_move[0], best_move[1], best_score, flag, depth]

        return best_move, best_score
