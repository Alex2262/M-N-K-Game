"""
Resizable tic-tac-toe type game with ai
by: alex
"""

import pygame
import cProfile
from pygame.locals import *
from badbot import *
from cache_clearer import kill_numba_cache
from mmbot import *
from abpbot import *
from ttbot import *
from mainbotv1 import MainAiV1
from mainbotv2 import MainAiV2
from mainbotv3 import MainAiV3
import v4.mainbotv4 as m4


def main():
    global basicFont, possibleMoves
    pygame.init()
    display = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    basicFont = pygame.font.SysFont('lucidagrande', BASICFONTSIZE)

    main_board = get_starting_board()
    move_archive = []
    create_pst()

    move = 0
    turn = 1  # 1 = O, 2 = X
    possibleMoves = find_moves(main_board)

    # run np functions first for njit
    test_board = numpy.array(main_board)
    np_test_for_win(test_board, WINNEED)
    np_test_for_win_enhanced(test_board, WINNEED, 0, 0)
    np_evaluate(test_board, turn)
    np_evaluate_neg(test_board, turn)
    np_tt_hash(test_board)
    np_find_moves(test_board)
    np_find_ordered_moves(test_board)
    np_find_noisy_moves(test_board, turn)

    main_ai_list = [MainAiV1(), MainAiV2(), MainAiV3(), m4.MainAiV4()]

    m4.compile_functions(main_ai_list[3], test_board)

    # run = 0 then play the game, run = 1 then review the game, run = 2 quit.
    run = 0
    while run == 0:
        draw_board(display, main_board)

        test_result = test_for_win(main_board)
        if test_result:
            print("player", test_result, "wins")
            if len(move_archive) > 0:
                print(np_test_for_win_enhanced(numpy.array(main_board), WINNEED,
                                               move_archive[-1][0],
                                               move_archive[-1][1]
                                               ))
            else:
                print(np_test_for_win_enhanced(numpy.array(main_board), WINNEED, -1, -1))
            run = 1
        elif len(possibleMoves) == 0:
            print("draw - no moves left")
            run = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = 2

            elif event.type == MOUSEBUTTONUP:
                # find the position of the square that was clicked
                spot_x, spot_y = get_spot_clicked(event.pos[0], event.pos[1])
                if (spot_x, spot_y) != (-1, -1):
                    if main_board[spot_y][spot_x] == 0:
                        move += 1
                        main_board[spot_y][spot_x] = turn
                        move_archive.append([spot_y, spot_x])
                        possibleMoves.remove([spot_y, spot_x])

                        if turn == 1:
                            turn = 2
                        else:
                            turn = 1

            elif event.type == pygame.KEYUP:
                # Check for certain key presses
                if event.key == pygame.K_b:  # activate bad bot algorithm
                    opp = 1 if turn == 2 else 2

                    reset_bb_nc()
                    start = timeit.default_timer()
                    chosen_move = bad_bot(copy.deepcopy(main_board),
                                          copy.deepcopy(possibleMoves), turn, opp, 0)
                    stop = timeit.default_timer()
                    print(stop-start, get_bb_nc())

                    if not chosen_move:
                        chosen_move = random.choice(possibleMoves)
                        print("Bad Bot:", chosen_move, "Losing")
                    elif chosen_move[1] == "choice":
                        chosen_move = random.choice(chosen_move[0])
                        print("Bad Bot:", chosen_move, "Choice")
                    else:
                        chosen_move = chosen_move[0]
                        print("Bad Bot:", chosen_move, "Forcing")

                    move += 1

                    main_board[chosen_move[0]][chosen_move[1]] = turn
                    possibleMoves.remove([chosen_move[0], chosen_move[1]])

                    move_archive.append(chosen_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_m:  # activate miniMax algorithm
                    move += 1

                    reset_mm_nc()
                    start = timeit.default_timer()
                    returned = minimax(main_board, turn, 0)
                    stop = timeit.default_timer()
                    print(stop-start, get_mm_nc())

                    best_move, best_score = list(returned[0]), returned[1]

                    print("MiniMax:", best_move, best_score)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_a:  # activate Alpha-beta pruned minimax algorithm
                    move += 1

                    reset_abp_nc()
                    start = timeit.default_timer()
                    returned = abp_minimax(main_board, turn, MIN, MAX, 0)
                    stop = timeit.default_timer()
                    print(stop-start, get_abp_nc())

                    best_move, best_score = list(returned[0]), returned[1]

                    print("ABP MiniMax:", best_move, best_score)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_t:  # activate TT Alpha-beta pruned minimax algorithm
                    move += 1

                    reset_tt_nc()
                    start = timeit.default_timer()
                    if ITERATIVEDEEPENING:
                        returned = tt_id(copy.deepcopy(main_board), turn)
                    else:
                        returned = tt_minimax(main_board, turn, MIN, MAX, TTD)
                    stop = timeit.default_timer()
                    print(stop-start, get_tt_nc())

                    best_move, best_score = list(returned[0]), returned[1]

                    print("TT MiniMax:", best_move, best_score)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_e:  # evaluate position
                    print(np_evaluate(numpy.array(main_board), turn) / 100)

                if event.key == pygame.K_1:
                    main_ai_list[0].node_count = 0
                    returned = main_ai_list[0].iterative_deepening(copy.deepcopy(main_board), turn)

                    best_move, best_score = list(returned[0]), returned[1]
                    print("MainBotV1:", best_move, best_score/100)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_2:
                    main_ai_list[1].node_count = 0
                    returned = main_ai_list[1].iterative_deepening(copy.deepcopy(main_board), turn)

                    best_move, best_score = list(returned[0]), returned[1]
                    print("MainBotV2:", best_move, best_score/100)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_3:
                    main_ai_list[2].node_count = 0
                    returned = main_ai_list[2].iterative_deepening(
                        copy.deepcopy(main_board), turn,
                        move_archive[-1] if len(move_archive) > 0 else [-1, -1])

                    best_move, best_score = list(returned[0]), returned[1]
                    print("MainBotV3:", best_move, best_score/100)
                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_4:
                    returned = m4.iterative_deepening(
                        main_ai_list[3],
                        copy.deepcopy(main_board), turn,
                        move_archive[-1] if len(move_archive) > 0 else [-1, -1])

                    print(returned)
                    best_move, best_score = list(returned[0]), returned[1]

                    print("MainBotV4:", best_move, best_score / 100)
                    wdl = max(0, min(100, 10 + pow(abs(best_score / 100) / 0.8, 1.7)))

                    if best_score > 0:
                        print(f"W:{round(wdl, 2)}% D:{round((100 - wdl) * 0.9, 2)}% L:{round((100 - wdl) * 0.1, 2)}%")
                    elif best_score == 0:
                        print(f"W:1% D:98% L:1%")
                    else:
                        print(f"W:{round((100 - wdl) * 0.1, 2)}% D:{round((100 - wdl) * 0.9, 2)}% L:{round(wdl, 2)}%")

                    main_board[best_move[0]][best_move[1]] = turn
                    possibleMoves.remove([best_move[0], best_move[1]])

                    move_archive.append(best_move)
                    turn = 1 if turn == 2 else 2

                if event.key == pygame.K_0:
                    '''returned = tt_id(copy.deepcopy(main_board), turn)
                    print("TT MiniMax:", returned[0], returned[1]/100)'''

                    '''returned = main_ai_list[0].iterative_deepening(copy.deepcopy(main_board), turn)
                    print("MainBotV1:", returned[0], returned[1]/100)'''

                    returned = main_ai_list[1].iterative_deepening(copy.deepcopy(main_board), turn)
                    print("MainBotV2:", returned[0], returned[1]/100)

                    returned = main_ai_list[2].iterative_deepening(
                        copy.deepcopy(main_board), turn,
                        move_archive[-1] if len(move_archive) > 0 else [-1, -1])
                    print("MainBotV3:", returned[0], returned[1]/100)
                if event.key == pygame.K_p:
                    if len(move_archive) > 0:
                        print(np_test_for_win_enhanced(numpy.array(main_board), WINNEED,
                                                       move_archive[-1][0],
                                                       move_archive[-1][1]
                                                       ))
                    else:
                        print(np_test_for_win_enhanced(numpy.array(main_board), WINNEED, -1, -1))

        pygame.display.update()
    if run > 0:
        pygame.quit()
    else:
        print(move)
        print(main_board)
        print(len(main_board))
        while run == 1:
            # when run == 1 you can review the game.
            draw_board(display, move_archive[move])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = 2

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        if move != 0:
                            move -= 1
                    elif event.key == pygame.K_RIGHT:
                        if move != len(move_archive) - 1:
                            move += 1

            pygame.display.update()

    pygame.quit()


def get_topleft_of_square(square_x, square_y):
    # get the topleft coordinate of a square
    left = XMARGIN + (square_x * SQUARESIZE) + (square_x - BORDERSIZE)
    top = YMARGIN + (square_y * SQUARESIZE) + (square_y - BORDERSIZE)
    return left, top


def get_spot_clicked(x, y):
    # from the x & y pixel coordinates, get the x & y board coordinates
    for square_y in range(BOARDHEIGHT):
        for square_x in range(BOARDWIDTH):
            left, top = get_topleft_of_square(square_x, square_y)
            square_rect = pygame.Rect(left, top, SQUARESIZE, SQUARESIZE)
            if square_rect.collidepoint(x, y):
                return square_x, square_y
    return -1, -1


def draw_square(display, square_x, square_y, turn):
    # draw a square
    left, top = get_topleft_of_square(square_y, square_x)
    pygame.draw.rect(display, SQUARECOLOR, (left + BORDERSIZE, top + BORDERSIZE, SQUARESIZE, SQUARESIZE))
    text_surf = basicFont.render(turn, True, TEXTCOLOR)
    text_rect = text_surf.get_rect()
    text_rect.center = left + int(SQUARESIZE / 2) + BORDERSIZE, top + int(SQUARESIZE / 2) + BORDERSIZE
    display.blit(text_surf, text_rect)


def draw_board(display, board):
    # draw the board
    display.fill((180, 180, 180))
    for i in range(BOARDHEIGHT):
        for j in range(BOARDWIDTH):
            if board[i][j] == 0:
                draw_square(display, i, j, "")
            elif board[i][j] == 1:
                draw_square(display, i, j, "O")
            elif board[i][j] == 2:
                draw_square(display, i, j, "X")


def clear_cache():
    kill_numba_cache()


if __name__ == "__main__":
    # clear_cache()
    main()
