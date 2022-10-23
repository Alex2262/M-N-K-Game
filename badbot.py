
import copy
from board import *

bb_node_count = 0


def get_bb_nc():
    return bb_node_count


def reset_bb_nc():
    global bb_node_count
    bb_node_count = 0


def bad_bot(board, moves, plr, opp, depth):
    curr_best = copy.deepcopy(moves)
    if len(moves) == 0:
        return "none"

    global bb_node_count
    bb_node_count += 1

    for i in range(len(moves)):
        test_board = copy.deepcopy(board)
        test_board[moves[i][0]][moves[i][1]] = plr

        test_result = test_for_win(test_board)
        if test_result == plr:
            return [moves[i], "forcing"]
        elif depth < BBD:
            new_moves = copy.deepcopy(moves)
            new_moves.remove(moves[i])
            chosen = bad_bot(test_board, new_moves, opp, plr, depth + 1)
            if chosen == "none":
                continue
            elif chosen:
                if chosen[1] == "forcing":
                    curr_best.remove(moves[i])
            else:
                return [moves[i], "forcing"]
    if len(curr_best) == 0:
        return False
    return [curr_best, "choice"]
