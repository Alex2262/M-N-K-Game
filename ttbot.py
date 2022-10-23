
import random
import timeit

from board import *


TRANSPOSITION_TABLE = {}
WIN_TABLE = {}

tt_start_time = 0
tt_node_count = 0


def get_tt_nc():
    return tt_node_count


def reset_tt_nc():
    global tt_node_count
    tt_node_count = 0


def tt_id(board, plr):
    global tt_start_time, tt_node_count

    node_sum = 0
    tt_start_time = timeit.default_timer()
    best_return = []
    for curr_depth in range(1, TTMD+1):
        tt_node_count = 0
        returned = tt_minimax(board, plr, MIN, MAX, curr_depth)
        node_sum += tt_node_count
        if returned[0] != "UNFINISHED":
            best_return = returned
            print(curr_depth, timeit.default_timer()-tt_start_time, best_return, tt_node_count, node_sum)
        else:
            print(curr_depth, "UNFINISHED", tt_node_count, node_sum)
            return best_return
    return best_return


def tt_minimax(board, plr, alpha, beta, depth):
    global tt_start_time
    if ITERATIVEDEEPENING and timeit.default_timer()-tt_start_time >= MAXTIME:
        return "UNFINISHED", 0

    alpha_org = alpha

    tt_code = tt_hash(board)
    pv_move = []
    if tt_code in TRANSPOSITION_TABLE:
        tt_entry = TRANSPOSITION_TABLE[tt_code]
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

    global tt_node_count
    tt_node_count += 1

    moves = find_moves(board)
    #random.shuffle(moves)

    if pv_move:
        moves.insert(0, moves.pop(moves.index(pv_move)))

    # if drawn
    if len(moves) == 0:
        return "none", 0

    # maximizing
    if plr == 1:

        best_score = MIN
        best_move = []
        for movePair in moves:
            y, x = movePair[0], movePair[1]
            board[y][x] = plr

            wt_code = tt_hash(board)
            if wt_code in WIN_TABLE:
                test_result = WIN_TABLE[wt_code]
            else:
                test_result = test_for_win(board)
                WIN_TABLE[wt_code] = test_result

            if test_result == plr:
                evaluation = 1000 + TTD + depth
                if evaluation > best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth > 0:
                returned = tt_minimax(board, 2, alpha, beta, depth - 1)
                if returned[0] == "UNFINISHED":
                    return returned
                return_eval = returned[1]
                if return_eval > best_score:
                    best_score = return_eval
                    best_move = movePair
            else:
                evaluation = evaluate(board, 1)
                if evaluation > best_score:
                    best_score = evaluation
                    best_move = movePair

            board[y][x] = 0
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break

        tt_store(TRANSPOSITION_TABLE, board, alpha_org, beta,
                 best_move, best_score, depth)

        return best_move, best_score

    # minimizing
    else:
        best_score = MAX
        best_move = []
        for movePair in moves:
            y, x = movePair[0], movePair[1]
            board[y][x] = plr

            wt_code = tt_hash(board)
            if wt_code in WIN_TABLE:
                test_result = WIN_TABLE[wt_code]
            else:
                test_result = test_for_win(board)
                WIN_TABLE[wt_code] = test_result

            if test_result == plr:
                evaluation = -1000 - depth
                if evaluation < best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth > 0:
                returned = tt_minimax(board, 1, alpha, beta, depth - 1)
                if returned[0] == "UNFINISHED":
                    return returned
                return_eval = returned[1]
                if return_eval < best_score:
                    best_score = return_eval
                    best_move = movePair
            else:
                evaluation = evaluate(board, 1)
                if evaluation < best_score:
                    best_score = evaluation
                    best_move = movePair

            board[y][x] = 0
            beta = min(beta, best_score)

            if beta <= alpha:
                break

        tt_store(TRANSPOSITION_TABLE, board, alpha_org, beta,
                 best_move, best_score, depth)

        return best_move, best_score
