
import copy
from board import *

bb_node_count = 0


def get_bb_nc():
    return bb_node_count


def reset_bb_nc():
    global bb_node_count
    bb_node_count = 0


def bad_bot(board, moves, plr, opp, depth):

    # Make a copy of the current moves to prevent referencing
    # We will continue removing moves from this list until we have a list of the best moves,
    # which we will return. The method that calls this can choose from any of the best moves.
    curr_best = copy.deepcopy(moves)

    # Return none if there are no moves to check.
    if len(moves) == 0:
        return "none"

    # Keep track of nodes searched
    global bb_node_count
    bb_node_count += 1

    # Loop through each move
    for i in range(len(moves)):

        # Copy the board to prevent referencing and changing the actual board.
        # This allows us to make moves to our own copy of the board, but doing this copy is slow.
        test_board = copy.deepcopy(board)
        test_board[moves[i][0]][moves[i][1]] = plr  # Make the move

        test_result = test_for_win(test_board)  # See if a win has been reached

        # Return the current move and detail it as a forcing move if we have won
        if test_result == plr:
            return [moves[i], "forcing"]

        # Otherwise, continue searching recursively as long as we are allowed.
        elif depth < BBD:
            new_moves = copy.deepcopy(moves)  # Make a new copy of the current moves for the child node
            new_moves.remove(moves[i])  # Remove the current move from the moves searched

            # Search recursively now, with the players switched and depth increased.
            # Chosen is the return value, which could be a boolean (only False), or a list.
            chosen = bad_bot(test_board, new_moves, opp, plr, depth + 1)

            # If there are no moves left in the child node, this is effectively a draw.
            # Continue searching other moves.
            if chosen == "none":
                continue

            # If the resulting value was not False
            elif chosen:
                # If the move was forcing, then that means the opponent was winning.
                # We do not want to return this move now.
                if chosen[1] == "forcing":
                    curr_best.remove(moves[i])

            # The resulting value was False, implying that the child node had no good moves to play,
            # So we are now winning, and we return "forcing".
            else:
                return [moves[i], "forcing"]

    # If we have no best moves, then return False.
    if len(curr_best) == 0:
        return False

    # Return a list containing a list of the best moves, paired with a comment stating that our move
    # can be a choice, and that it was not forced.
    return [curr_best, "choice"]
