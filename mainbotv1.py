

import timeit
import cProfile
from board import *


class MainAiV1:

    def __init__(self):
        self.TRANSPOSITION_TABLE = {}
        self.WIN_TABLE = {}

        self.max_depth = 9
        self.min_stop_depth = 1
        self.curr_depth = self.max_depth

        self.start_time = 0
        self.node_count = 0

    def iterative_deepening(self, board, plr):
        self.start_time = timeit.default_timer()

        np_board = numpy.array(board)
        node_sum = 0
        best_return = []
        for running_depth in range(1, self.max_depth + 1):
            self.node_count = 0
            self.curr_depth = running_depth

            returned = self.search(np_board, plr, MIN, MAX, running_depth)

            node_sum += self.node_count
            if returned[0] != "UNFINISHED":
                best_return = [returned[0], returned[1]]
                print(running_depth, timeit.default_timer() - self.start_time,
                      best_return[0], best_return[1] / 100, self.node_count, node_sum)
            else:
                print(running_depth, "UNFINISHED", self.node_count, node_sum)
                return best_return
        return best_return

    def search(self, board, plr, alpha, beta, depth):
        if timeit.default_timer() - self.start_time >= MAXTIME and self.curr_depth >= self.min_stop_depth:
            return "UNFINISHED", 0

        alpha_org = alpha

        tt_code = np_tt_hash(board)
        pv_move = []
        if tt_code in self.TRANSPOSITION_TABLE:
            tt_entry = self.TRANSPOSITION_TABLE[tt_code]
            if tt_entry[3] >= depth:
                if tt_entry[2] == 0:  # checking for Exact
                    return tt_entry[0], tt_entry[1]
                elif tt_entry[2] == -1:  # lower bound
                    alpha = max(alpha, tt_entry[1])
                elif tt_entry[2] == 1:  # upper bound
                    beta = min(beta, tt_entry[1])

                if alpha >= beta:
                    return tt_entry[0], tt_entry[1]
            else:
                pv_move = tt_entry[0]

        moves = find_moves(board)

        if pv_move:
            moves.insert(0, moves.pop(moves.index(pv_move)))

        # if drawn
        if len(moves) == 0:
            return "none", 0

        # maximizing
        if plr == 1:

            best_score = -100000
            best_move = []
            for movePair in moves:
                y, x = movePair[0], movePair[1]
                board[y, x] = plr

                wt_code = np_tt_hash(board)
                if wt_code in self.WIN_TABLE:
                    test_result = self.WIN_TABLE[wt_code]
                else:
                    test_result = np_test_for_win(board, WINNEED)
                    self.WIN_TABLE[wt_code] = test_result
                if test_result == plr:
                    evaluation = 10000 + depth
                    if evaluation > best_score:
                        best_score = evaluation
                        best_move = movePair

                elif depth > 0:
                    self.node_count += 1
                    returned = self.search(board, 2, alpha, beta, depth - 1)
                    if returned[0] == "UNFINISHED":
                        return returned
                    return_eval = returned[1]
                    if return_eval > best_score:
                        best_score = return_eval
                        best_move = movePair
                else:
                    evaluation = np_evaluate(board, 2)
                    if evaluation > best_score:
                        best_score = evaluation
                        best_move = movePair

                board[y, x] = 0
                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

            tt_store(self.TRANSPOSITION_TABLE, board, alpha_org, beta,
                     best_move, best_score, depth)

            return best_move, best_score

        # minimizing
        else:
            best_score = 100000
            best_move = []
            for movePair in moves:
                y, x = movePair[0], movePair[1]
                board[y, x] = plr

                wt_code = np_tt_hash(board)
                if wt_code in self.WIN_TABLE:
                    test_result = self.WIN_TABLE[wt_code]
                else:
                    test_result = np_test_for_win(board, WINNEED)
                    self.WIN_TABLE[wt_code] = test_result
                if test_result == plr:
                    evaluation = -10000 - depth
                    if evaluation < best_score:
                        best_score = evaluation
                        best_move = movePair

                elif depth > 0:
                    self.node_count += 1
                    returned = self.search(board, 1, alpha, beta, depth - 1)
                    if returned[0] == "UNFINISHED":
                        return returned
                    return_eval = returned[1]
                    if return_eval < best_score:
                        best_score = return_eval
                        best_move = movePair
                else:
                    evaluation = np_evaluate(board, 1)
                    if evaluation < best_score:
                        best_score = evaluation
                        best_move = movePair

                board[y, x] = 0
                beta = min(beta, best_score)

                if beta <= alpha:
                    break

            tt_store(self.TRANSPOSITION_TABLE, board, alpha_org, beta,
                     best_move, best_score, depth)

            return best_move, best_score
