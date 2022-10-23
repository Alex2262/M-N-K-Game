import time
from math import sqrt

import numba
from numba import int32, float64
from numba.experimental import jitclass

from v4.boardv4 import *

NUMBA_HASH_TYPE = numba.from_dtype(numpy.dtype(
    [("key", numpy.uint64), ("score", numpy.int32), ("flag", numpy.uint8),
     ("move_from", numpy.uint16), ("move_to", numpy.uint16),
     ("depth", numpy.int8)]
))

HASH_FLAG_EXACT, HASH_FLAG_ALPHA, HASH_FLAG_BETA = (0, 1, 2)

spec = [
    ('max_depth', int32),
    ('min_stop_depth', int32),
    ('curr_depth', int32),
    ('start_time', float64),
    ('node_count', int32),
    ('stopped', numba.boolean),
    ('pst', numba.int32[:, :]),
    ('hash_key', numba.uint64),
    ("transposition_table", NUMBA_HASH_TYPE[:])
]


@jitclass(spec=spec)
class MainAiV4:

    def __init__(self):

        self.max_depth = 15
        self.min_stop_depth = 1
        self.curr_depth = 0

        self.start_time = 0
        self.node_count = 0

        self.stopped = False

        self.pst = numpy.zeros((BOARDHEIGHT, BOARDWIDTH), dtype=numpy.int32)
        for i in range(BOARDHEIGHT):
            for j in range(BOARDWIDTH):
                self.pst[i][j] = int((YMID / 2 - math.floor(abs(YMID - i))) * 12
                                     + (XMID / 2 - math.floor(abs(XMID - j))) * 12)

        self.hash_key = 0
        self.transposition_table = numpy.zeros(MAX_HASH_SIZE, dtype=NUMBA_HASH_TYPE)


@njit(cache=True)
def get_time():
    with numba.objmode(time_amt=numba.float64):
        time_amt = time.time()

    return time_amt


@njit(cache=True)
def check_time(engine):
    if get_time() - engine.start_time >= MAXTIME and engine.curr_depth >= engine.min_stop_depth:
        engine.stopped = True


@njit  # (numba.int64[:](MainAiV4.class_type.instance_type, int32[:, :], int32, int32, int32, int32, int32, int32))
def negamax(engine, board, plr, alpha, beta, last_y, last_x, depth):

    index = engine.hash_key % MAX_HASH_SIZE

    test_result = check_win(board, WINNEED, last_y, last_x)
    if test_result:
        return -1, -1, -10000 - depth

    if depth == 0:
        return -1, -1, evaluate(engine, board, plr)

    moves = get_sorted_moves(engine, board)
    if len(moves) == 0:
        return -1, -1, 0

    if (engine.node_count & 255) == 0:
        check_time(engine)

    if engine.stopped:
        return -2, -2, 0

    pv_move = (-1, -1)

    entry = engine.transposition_table[index]

    if entry.key == engine.hash_key:
        if entry.depth >= depth:
            if entry.flag == HASH_FLAG_EXACT:
                return entry.move_from, entry.move_to, entry.score
            if entry.flag == HASH_FLAG_ALPHA and entry.score <= alpha:
                return entry.move_from, entry.move_to, entry.score
            if entry.flag == HASH_FLAG_BETA and entry.score >= beta:
                return entry.move_from, entry.move_to, entry.score

        pv_move = (entry.move_from, entry.move_to)

    tt_hash_flag = HASH_FLAG_ALPHA

    if pv_move != (-1, -1):
        moves.insert(0, moves.pop(moves.index(pv_move)))

    pv_node = alpha != beta - 1

    engine.node_count += 1

    best_move = (-1, -1)
    best_score = -100000
    legal_moves = 0

    for move in moves:

        board[move[0]][move[1]] = plr
        engine.hash_key ^= BOARD_HASH_KEYS[plr - 1][move[0]][move[1]]

        reduction = 0

        # Late move reductions
        if legal_moves >= 6 + sqrt(BOARDHEIGHT * BOARDWIDTH) / 3 and depth >= 3:
            reduction += 1

            if pv_node:
                reduction -= 1

            if legal_moves >= 18 + sqrt(BOARDHEIGHT * BOARDWIDTH) / 2:
                reduction += 1
            if legal_moves >= 36 + sqrt(BOARDHEIGHT * BOARDWIDTH):
                reduction += 1
            if legal_moves >= (BOARDHEIGHT * BOARDWIDTH) * 0.6:
                reduction += 1
            if legal_moves >= (BOARDHEIGHT * BOARDWIDTH) * 0.8:
                reduction += 1

            reduction = min(depth - 1, reduction)

        # PVS
        if legal_moves == 0:
            returned = negamax(engine, board, ((plr - 1) ^ 1) + 1, -beta, -alpha, move[0], move[1], depth - 1)
        else:
            returned = negamax(engine, board, ((plr - 1) ^ 1) + 1,
                               -alpha - 1, -alpha, move[0], move[1], depth - reduction - 1)

        if -returned[2] > alpha and reduction:
            returned = negamax(engine, board, ((plr - 1) ^ 1) + 1, -alpha - 1, -alpha, move[0], move[1],
                               depth - 1)

        if -returned[2] > alpha and legal_moves:
            returned = negamax(engine, board, ((plr - 1) ^ 1) + 1, -beta, -alpha, move[0], move[1],
                               depth - 1)

        engine.hash_key ^= BOARD_HASH_KEYS[plr - 1][move[0]][move[1]]
        board[move[0]][move[1]] = 0

        if engine.stopped:
            return -2, -2, 0

        return_eval = -returned[2]
        if return_eval > best_score:
            best_score = return_eval
            best_move = move

        if return_eval > alpha:
            alpha = return_eval
            tt_hash_flag = HASH_FLAG_EXACT

            if alpha >= beta:
                break

        legal_moves += 1

    if entry.key != engine.hash_key or depth > entry.depth or tt_hash_flag == HASH_FLAG_EXACT:
        entry.key = engine.hash_key
        entry.depth = depth
        entry.flag = tt_hash_flag
        entry.score = best_score
        entry.move_from = best_move[0]
        entry.move_to = best_move[1]

    return best_move[0], best_move[1], best_score


def iterative_deepening(engine, board, plr, last_move):

    engine.node_count = 0
    engine.stopped = False
    engine.start_time = time.time()

    np_board = numpy.array(board)
    engine.hash_key = numba.uint64(generate_hash_key(np_board))

    node_sum = 0
    best_return = []

    for running_depth in range(1, engine.max_depth + 1):
        engine.node_count = 0
        engine.curr_depth = running_depth

        returned = negamax(engine, np_board, plr, MIN, MAX, last_move[0], last_move[1], engine.curr_depth + 1)

        node_sum += engine.node_count
        if returned[0] != -2:
            best_return = [[returned[0], returned[1]], returned[2] if plr == 1 else -returned[1]]
            print(engine.curr_depth, time.time() - engine.start_time,
                  best_return[0], best_return[1] / 100, engine.node_count, node_sum)
        else:
            print(engine.curr_depth, "UNFINISHED", engine.node_count, node_sum)
            return best_return

    return best_return


def compile_functions(engine, board):
    negamax(engine, board, 1, MIN, MAX, -1, -1, 1)
