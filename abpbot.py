
import random
from board import *

abp_node_count = 0


def get_abp_nc():
    return abp_node_count


def reset_abp_nc():
    global abp_node_count
    abp_node_count = 0


def abp_minimax(board, plr, alpha, beta, depth):
    moves = find_moves(board)
    random.shuffle(moves)  # random move order improves branches pruned.

    # if drawn
    if len(moves) == 0:
        return "none", 0

    global abp_node_count
    abp_node_count += 1

    # maximizing
    if plr == 1:

        best_score = MIN
        best_move = []
        for movePair in moves:
            y, x = movePair[0], movePair[1]
            board[y][x] = plr

            test_result = test_for_win(board)
            if test_result == plr:
                evaluation = 1000 + AMD - depth
                if evaluation > best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth < AMD:
                returned = abp_minimax(board, 2, alpha, beta, depth + 1)
                return_eval = returned[1]
                if return_eval > best_score:
                    best_score = return_eval
                    best_move = movePair
            else:
                evaluation = evaluate(board, 2)
                if evaluation > best_score:
                    best_score = evaluation
                    best_move = movePair

            board[y][x] = 0
            alpha = max(alpha, best_score)

            if beta <= alpha:
                break
        return best_move, best_score

    # minimizing
    else:
        best_score = MAX
        best_move = []
        for movePair in moves:
            y, x = movePair[0], movePair[1]
            board[y][x] = plr

            test_result = test_for_win(board)
            if test_result == plr:
                evaluation = -1000 - AMD + depth
                if evaluation < best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth < AMD:
                returned = abp_minimax(board, 1, alpha, beta, depth + 1)
                return_eval = returned[1]
                if return_eval < best_score:
                    best_score = return_eval
                    best_move = movePair
            else:
                evaluation = evaluate(board, 2)
                if evaluation < best_score:
                    best_score = evaluation
                    best_move = movePair

            board[y][x] = 0
            beta = min(beta, best_score)

            if beta <= alpha:
                break
        return best_move, best_score
