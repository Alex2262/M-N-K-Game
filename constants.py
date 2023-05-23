
BOARDWIDTH = 9  # amount of x-axis squares
BOARDHEIGHT = 9  # amount of y-axis squares
WINNEED = 5  # amount in a row to win

SQUARESIZE = 40
SQUARECOLOR = (200, 200, 200)
BORDERSIZE = 1

XMARGIN = (SQUARESIZE + BORDERSIZE) * 2
YMARGIN = (SQUARESIZE + BORDERSIZE) * 2

WINDOWWIDTH = BOARDWIDTH * SQUARESIZE + XMARGIN * 2
WINDOWHEIGHT = BOARDHEIGHT * SQUARESIZE + YMARGIN * 2

BBD = 9  # bad bot fixed depth
MMD = 9  # minimax fixed depth
AMD = 9  # alpha beta pruned mini fixed depth
TTD = 9  # transposition table + abp minimax fixed depth
TTMD = 9  # TT Iterative Deepening max depth

ITERATIVEDEEPENING = True
MAXTIME = 8

BASICFONTSIZE = 30
TEXTCOLOR = (140, 140, 140)

YMID = (BOARDHEIGHT - 1) / 2
XMID = (BOARDWIDTH - 1) / 2

MIN = -10000
MAX = 10000

MAX_HASH_SIZE = 1000000
