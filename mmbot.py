
from board import *

mm_node_count = 0


def get_mm_nc():
    return mm_node_count


def reset_mm_nc():
    global mm_node_count
    mm_node_count = 0


def minimax(board, plr, depth):
    moves = find_moves(board)

    # if drawn
    if len(moves) == 0:
        return "none", 0

    global mm_node_count
    mm_node_count += 1

    if plr == 1:
        best_score = MIN
        best_move = []
        for movePair in moves:
            y, x = movePair[0], movePair[1]
            board[y][x] = plr

            test_result = test_for_win(board)
            if test_result == plr:
                evaluation = 1000 + MMD - depth
                if evaluation > best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth < MMD:
                returned = minimax(board, 2, depth + 1)
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
                evaluation = -1000 - MMD + depth
                if evaluation < best_score:
                    best_score = evaluation
                    best_move = movePair
            elif depth < MMD:
                returned = minimax(board, 1, depth + 1)
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
        return best_move, best_score
