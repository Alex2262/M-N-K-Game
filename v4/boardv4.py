import math

import numba
import numpy
from numba import njit
from constants import *


spec = [
    ('pst', numba.int32[:, :])
]

traversal_increments = numpy.array(
    ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
)


BOARD_HASH_KEYS = numpy.random.randint(1, 2**64 - 1, size=(2, BOARDHEIGHT, BOARDWIDTH), dtype=numpy.uint64)


@njit(cache=True)
def generate_hash_key(board):
    code = numba.uint64(0)
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j]:
                code ^= BOARD_HASH_KEYS[board[i][j] - 1][i][j]

    return code


@njit(cache=True)
def get_sorted_moves(engine, board):
    moves = []
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j] == 0:
                moves.append((i, j))

    move_scores = numpy.zeros(len(moves))

    for i, move in enumerate(moves):
        move_scores[i] += engine.pst[move[0]][move[1]]

    zipped_moves = zip(move_scores, moves)
    ordered_moves = [move for _, move in sorted(zipped_moves, reverse=True)]
    return ordered_moves


@njit(cache=True)
def check_win(board, amt, last_y, last_x):
    if last_x == -1:
        return False

    color = board[last_y][last_x]
    for increment in traversal_increments:
        new_y = last_y
        new_x = last_x
        for i in range(amt - 1):  # excluding starting point
            new_y += increment[0]
            new_x += increment[1]
            if new_y >= BOARDHEIGHT or new_y < 0 or new_x >= BOARDWIDTH or new_x < 0 or board[new_y][new_x] != color:
                break
            if i == amt - 2:  # -2 for excluding starting point and because index starts at 0
                return True

    return False


@njit(cache=True)
def heuristic(engine, board, plr, y, x):
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

    return numba.int32(ssv - bsv + engine.pst[y, x])


@njit(cache=True)
def evaluate(engine, board, turn):
    o_score = 0
    x_score = 0

    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j] == 1:
                o_score += heuristic(engine, board, 1, i, j)
            elif board[i][j] == 2:
                x_score += heuristic(engine, board, 2, i, j)

    return numba.int32(-1 * (2 * turn - 3) * (o_score - x_score))
