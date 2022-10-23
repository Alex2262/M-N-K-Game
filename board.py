
import math
import numpy
from constants import *
from numba import njit


PST = []  # Piece Square Tables
NPPST = numpy.empty((BOARDHEIGHT, BOARDWIDTH))

traversal_increments = numpy.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
)


def get_starting_board():
    # create a blank board
    board = []
    for i in range(BOARDHEIGHT):
        row = []
        for j in range(BOARDWIDTH):
            row.append(0)
        board.append(row)
    return board


def create_pst():
    global NPPST
    for i in range(BOARDHEIGHT):
        row = []
        for j in range(BOARDWIDTH):
            row.append(int((YMID/2-math.floor(abs(YMID-i)))*12
                       + (XMID/2-math.floor(abs(XMID-j)))*12))
        PST.append(row)
        print(row)
    NPPST = numpy.array(PST)


def find_moves(board):
    moves = []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j] == 0:
                moves.append([i, j])
    return moves


@njit(cache=True)
def np_find_moves(board):
    moves = []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i, j] == 0:
                moves.append([i, j])
    return moves


@njit(cache=True)  # revise this to make faster?
def np_find_ordered_moves(board):
    moves = []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i, j] == 0:
                moves.append([i, j])
    move_scores = numpy.zeros(len(moves))
    for i, move in enumerate(moves):
        y, x = move[0], move[1]
        move_scores[i] += NPPST[y, x]
    zipped_moves = zip(move_scores, moves)
    ordered_moves = [move for _, move in sorted(zipped_moves, reverse=True)]
    return ordered_moves


@njit(cache=True)  # not really necessary since it's a 1 move extension — instead maybe finding threatening and winning moves
def np_find_noisy_moves(board, turn):
    moves = []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i, j] == 0:
                board[i, j] = turn
                if np_test_for_win(board, WINNEED) == turn:
                    moves.append([i, j])
                board[i, j] = 0
    return moves


def test_for_win(board):
    # calculate the diagonals, horizontals and verticals if there are WINNEED connected
    # loop through x axis

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):

            # topleft to bottom right
            for c in range(1, min(BOARDHEIGHT - i, BOARDWIDTH - j, WINNEED)):
                if board[i + c][j + c] == 0:
                    break
                elif board[i + c][j + c] != board[i + c - 1][j + c - 1]:
                    break
                elif c == WINNEED - 1:
                    return board[i + c][j + c]

            # bottom left to top right
            for c in range(1, min(i + 1, BOARDWIDTH - j, WINNEED)):
                if board[i - c][j + c] == 0:
                    break
                elif board[i - c][j + c] != board[i - c + 1][j + c - 1]:
                    break
                elif c == WINNEED - 1:
                    return board[i - c][j + c]

            # left to right
            for c in range(1, min(BOARDWIDTH - j, WINNEED)):
                if board[i][j + c] == 0:
                    break
                elif board[i][j + c] != board[i][j + c - 1]:
                    break
                elif c == WINNEED - 1:
                    return board[i][j + c]

            # top to bottom
            for c in range(1, min(BOARDHEIGHT - i, WINNEED)):
                if board[i + c][j] == 0:
                    break
                elif board[i + c][j] != board[i + c - 1][j]:
                    break
                elif c == WINNEED - 1:
                    return board[i + c][j]

    return False


@njit(cache=True)
def np_test_for_win(board, amt):
    # calculate the diagonals, horizontals and verticals if there are WINNEED connected
    # loop through x axis

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):

            # topleft to bottom right
            for c in range(1, min(BOARDHEIGHT - i, BOARDWIDTH - j, WINNEED)):
                if board[i + c, j + c] == 0:
                    break
                elif board[i + c, j + c] != board[i + c - 1, j + c - 1]:
                    break
                elif c == amt - 1:
                    return board[i + c, j + c]

            # bottom left to top right
            for c in range(1, min(i + 1, BOARDWIDTH - j, WINNEED)):
                if board[i - c, j + c] == 0:
                    break
                elif board[i - c, j + c] != board[i - c + 1, j + c - 1]:
                    break
                elif c == amt - 1:
                    return board[i - c, j + c]

            # left to right
            for c in range(1, min(BOARDWIDTH - j, WINNEED)):
                if board[i, j + c] == 0:
                    break
                elif board[i, j + c] != board[i, j + c - 1]:
                    break
                elif c == amt - 1:
                    return board[i, j + c]

            # top to bottom
            for c in range(1, min(BOARDHEIGHT - i, WINNEED)):
                if board[i + c, j] == 0:
                    break
                elif board[i + c, j] != board[i + c - 1, j]:
                    break
                elif c == amt - 1:
                    return board[i + c, j]

    return False


@njit(cache=True)
def np_test_for_win_enhanced(board, amt, last_y, last_x):
    if last_x == -1:
        return False

    color = board[last_y, last_x]
    for increment in traversal_increments:
        new_y = last_y
        new_x = last_x
        for i in range(amt-1):  # excluding starting point
            new_y += increment[0]
            new_x += increment[1]
            if new_y >= BOARDHEIGHT or new_y < 0 or new_x >= BOARDWIDTH or new_x < 0 or board[new_y][new_x] != color:
                break
            if i == amt-2:  # -2 for excluding starting point and because index starts at 0
                return True

    return False


def get_position_value_v2(board, plr, y, x):
    opp = 1 if plr == 2 else 2
    ssv = 0  # same square value
    bsv = 0  # bsv

    # up
    for i in range(1, BOARDHEIGHT):
        cssn = 0  # current num of same square
        if y - i < 0:
            break
        elif board[y - i][x] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i][x] == opp:
            bsv += 3 * cssn
            break
    # down
    for i in range(1, BOARDHEIGHT):
        cssn = 0
        if y + i > BOARDHEIGHT - 1:
            break
        elif board[y + i][x] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i][x] == opp:
            bsv += 3 * cssn
            break
    # left
    for i in range(1, BOARDWIDTH):
        cssn = 0
        if x - i < 0:
            break
        elif board[y][x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y][x - i] == opp:
            bsv += 3 * cssn
            break
    # right
    for i in range(1, BOARDWIDTH):
        cssn = 0
        if x + i > BOARDWIDTH - 1:
            break
        elif board[y][x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y][x + i] == opp:
            bsv += 3 * cssn
            break
    # left-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if min(y - i, x - i) < 0:
            break
        elif board[y - i][x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i][x - i] == opp:
            bsv += 3 * cssn
            break
    # right-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if y - i < 0 or x + i > BOARDWIDTH - 1:
            break
        elif board[y - i][x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i][x + i] == opp:
            bsv += 3 * cssn
            break
    # left-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if y + i > BOARDHEIGHT - 1 or x - i < 0:
            break
        elif board[y + i][x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i][x - i] == opp:
            bsv += 3 * cssn
            break
    # right-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if max(y + i, x + i) > min(BOARDHEIGHT, BOARDWIDTH) - 1:
            break
        elif board[y + i][x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i][x + i] == opp:
            bsv += 3 * cssn
            break

    return ssv - bsv + PST[y][x]


def get_position_value_v1(board, plr, y, x):
    opp = 1 if plr == 2 else 2
    same_squares = 0
    y_from_mid = int((YMID - math.floor(abs(YMID - y)))*10)
    x_from_mid = int((XMID - math.floor(abs(XMID - x)))*10)

    # up
    for i in range(1, BOARDHEIGHT):
        if y - i < 0:
            break
        elif board[y - i][x] == plr:
            same_squares += 10
        elif board[y - i][x] == opp:
            break
    # down
    for i in range(1, BOARDHEIGHT):
        if y + i > BOARDHEIGHT - 1:
            break
        elif board[y + i][x] == plr:
            same_squares += 10
        elif board[y + i][x] == opp:
            break
    # left
    for i in range(1, BOARDWIDTH):
        if x - i < 0:
            break
        elif board[y][x - i] == plr:
            same_squares += 10
        elif board[y][x - i] == opp:
            break
    # right
    for i in range(1, BOARDWIDTH):
        if x + i > BOARDWIDTH - 1:
            break
        elif board[y][x + i] == plr:
            same_squares += 10
        elif board[y][x + i] == opp:
            break
    # left-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        if min(y - i, x - i) < 0:
            break
        elif board[y - i][x - i] == plr:
            same_squares += 10
        elif board[y - i][x - i] == opp:
            break
    # right-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        if y - i < 0 or x + i > BOARDWIDTH - 1:
            break
        elif board[y - i][x + i] == plr:
            same_squares += 10
        elif board[y - i][x + i] == opp:
            break
    # left-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        if y + i > BOARDHEIGHT - 1 or x - i < 0:
            break
        elif board[y + i][x - i] == plr:
            same_squares += 10
        elif board[y + i][x - i] == opp:
            break
    # right-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        if max(y + i, x + i) > min(BOARDHEIGHT, BOARDWIDTH) - 1:
            break
        elif board[y + i][x + i] == plr:
            same_squares += 10
        elif board[y + i][x + i] == opp:
            break

    return same_squares + y_from_mid + x_from_mid


@njit(cache=True)
def np_heuristic2(board):
    # calculate the diagonals, horizontals and verticals if there are WINNEED connected
    # loop through x axis

    for i in range(BOARDHEIGHT):
        # left to right
        for c in range(1, min(BOARDWIDTH, WINNEED)):
            if board[i, c] == 0:
                break
            elif board[i, c] != board[i, c - 1]:
                break
            elif c == WINNEED - 1:
                return board[i, c]

        # topleft to bottom right
        for c in range(1, min(BOARDHEIGHT - i, BOARDWIDTH, WINNEED)):
            if board[i + c, c] == 0:
                break
            elif board[i + c, c] != board[i + c - 1, c - 1]:
                break
            elif c == WINNEED - 1:
                return board[i + c, c]

        # bottom left to top right
        for c in range(1, min(BOARDHEIGHT, BOARDWIDTH, WINNEED)):
            if board[BOARDHEIGHT - i - c, c] == 0:
                break
            elif board[BOARDHEIGHT - i - c, c] != board[BOARDHEIGHT - i - c + 1, c - 1]:
                break
            elif c == WINNEED - 1:
                return board[i - c, c]

    for i in range(BOARDWIDTH):
        # top to bottom
        for c in range(1, min(BOARDHEIGHT, WINNEED)):
            if board[c, i] == 0:
                break
            elif board[c, i] != board[c - 1, i]:
                break
            elif c == WINNEED - 1:
                return board[c, i]

        # topleft to bottom right
        for c in range(1, min(BOARDHEIGHT - i, BOARDWIDTH, WINNEED)):
            if board[c, i + c] == 0:
                break
            elif board[c, i + c] != board[c - 1, i + c - 1]:
                break
            elif c == WINNEED - 1:
                return board[c, i + c]

        # bottom left to top right
        for c in range(1, min(BOARDHEIGHT, BOARDWIDTH - i, WINNEED)):
            if board[BOARDHEIGHT - c, i + c] == 0:
                break
            elif board[BOARDHEIGHT - c, i + c] != board[BOARDHEIGHT - c + 1, i + c - 1]:
                break
            elif c == WINNEED - 1:
                return board[BOARDHEIGHT - c, i + c]

    return False


@njit(cache=True)
def np_heuristic(board, plr, turn, y, x):
    opp = 1 if plr == 2 else 2
    ssv = 0  # same square value
    bsv = 0  # bsv

    # up
    for i in range(1, BOARDHEIGHT):
        cssn = 0  # current num of same square
        if y - i < 0:
            break
        elif board[y - i, x] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i, x] == opp:
            bsv += 3 * cssn
            break
    # down
    for i in range(1, BOARDHEIGHT):
        cssn = 0
        if y + i > BOARDHEIGHT - 1:
            break
        elif board[y + i, x] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i, x] == opp:
            bsv += 3 * cssn
            break

    # left
    for i in range(1, BOARDWIDTH):
        cssn = 0
        if x - i < 0:
            break
        elif board[y, x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y, x - i] == opp:
            bsv += 3 * cssn
            break
    # right
    for i in range(1, BOARDWIDTH):
        cssn = 0
        if x + i > BOARDWIDTH - 1:
            break
        elif board[y, x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y, x + i] == opp:
            bsv += 3 * cssn
            break

    # left-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if min(y - i, x - i) < 0:
            break
        elif board[y - i, x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i, x - i] == opp:
            bsv += 3 * cssn
            break
    # right-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if max(y + i, x + i) > min(BOARDHEIGHT, BOARDWIDTH) - 1:
            break
        elif board[y + i, x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i, x + i] == opp:
            bsv += 3 * cssn
            break

    # right-up
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if y - i < 0 or x + i > BOARDWIDTH - 1:
            break
        elif board[y - i, x + i] == plr:
            cssn += 1
            ssv += 16
        elif board[y - i, x + i] == opp:
            bsv += 3 * cssn
            break
    # left-down
    for i in range(1, min(BOARDHEIGHT, BOARDWIDTH)):
        cssn = 0
        if y + i > BOARDHEIGHT - 1 or x - i < 0:
            break
        elif board[y + i, x - i] == plr:
            cssn += 1
            ssv += 16
        elif board[y + i, x - i] == opp:
            bsv += 3 * cssn
            break

    return ssv - bsv + NPPST[y, x]


def evaluate(board, version):
    o_score = 0
    x_score = 0

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j] == 1:
                o_score += eval("get_position_value_v"+str(version)+"(board, 1, i, j)")
            elif board[i][j] == 2:
                x_score += eval("get_position_value_v"+str(version)+"(board, 2, i, j)")

    return o_score - x_score


@njit(cache=True)
def np_evaluate(board, turn):
    o_score = 0
    x_score = 0

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i, j] == 1:
                o_score += np_heuristic(board, 1, turn, i, j)
            elif board[i, j] == 2:
                x_score += np_heuristic(board, 2, turn, i, j)

    return o_score - x_score


@njit(cache=True)
def np_evaluate_neg(board, turn):
    o_score = 0
    x_score = 0

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i, j] == 1:
                o_score += np_heuristic(board, 1, turn, i, j)
            elif board[i, j] == 2:
                x_score += np_heuristic(board, 2, turn, i, j)

    return o_score - x_score if turn == 1 else x_score - o_score


def tt_hash(board):
    code = ""
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            code += str(board[i][j])
    return code


@njit(cache=True)
def np_tt_hash(board):
    code = ""
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            code += str(board[i, j])
    return code


def tt_store(table, board, alpha, beta, move, score, depth):
    if score <= alpha:
        flag = 1  # Upper bound
    elif score >= beta:
        flag = -1  # Lower bound
    else:
        flag = 0  # Exact

    table[tt_hash(board)] = [move, score, flag, depth]
