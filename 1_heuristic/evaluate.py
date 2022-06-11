from enum import Enum

__all__ = ["evaluate", "Pattern"]


class Pattern(Enum):
    ONE = 10
    TWO = 100
    THREE = 1000
    FOUR = 100000
    FIVE = 10000000
    BLOCKED_ONE = 1
    BLOCKED_TWO = 15
    BLOCKED_THREE = 150
    BLOCKED_FOUR = 15000

    # print(Pattern.BLOCKED_FOUR.value)


lived_four = [[0, 1, 1, 1, 1, 0]]

pattern_map = {}

for four_p in lived_four:
    tmp4 = four_p.copy()
    pattern_map[tuple(tmp4)] = Pattern.FOUR
    for i3 in range(len(tmp4)):
        if tmp4[i3] == 1:
            tmp3 = tmp4.copy()
            tmp3[i3] = 0
            pattern_map[tuple(tmp3)] = Pattern.THREE
            for i2 in range(len(tmp3)):
                if tmp3[i2] == 1:
                    tmp2 = tmp3.copy()
                    tmp2[i2] = 0
                    pattern_map[tuple(tmp2)] = Pattern.TWO
                    for i1 in range(len(tmp2)):
                        if tmp2[i1] == 1:
                            tmp1 = tmp2.copy()
                            tmp1[i1] = 0
                            pattern_map[tuple(tmp1)] = Pattern.ONE

print(pattern_map)

blocked_four = [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]]
for blocked_four_p in blocked_four:
    tmp4 = blocked_four_p.copy()
    pattern_map[tuple(tmp4)] = Pattern.BLOCKED_FOUR
    for i3 in range(len(tmp4)):
        if tmp4[i3] == 1:
            tmp3 = tmp4.copy()
            tmp3[i3] = 0
            pattern_map[tuple(tmp3)] = Pattern.BLOCKED_THREE
            for i2 in range(len(tmp3)):
                if tmp3[i2] == 1:
                    tmp2 = tmp3.copy()
                    tmp2[i2] = 0
                    pattern_map[tuple(tmp2)] = Pattern.BLOCKED_TWO
                    for i1 in range(len(tmp2)):
                        if tmp2[i1] == 1:
                            tmp1 = tmp2.copy()
                            tmp1[i1] = 0
                            pattern_map[tuple(tmp1)] = Pattern.BLOCKED_ONE

for k in pattern_map:
    print(k, pattern_map[k])


def get4lines(board, x, y, actor):
    h, w = board.shape
    lines = []

    line = []  # -
    for i in range(1, 5):
        if x - i < 0 or board[y][x - i] == -actor:
            break
        line.insert(0, board[y][x - i])

    for i in range(0, 5):
        if x + i == w or board[y][x + i] == -actor:
            break
        line.append(board[y][x + i])
    lines.append(line)

    line = []  # |
    for i in range(1, 5):
        if y - i < 0 or board[y - i][x] == -actor:
            break
        line.insert(0, board[y - i][x])

    for i in range(0, 5):
        if y + i == h or board[y + i][x] == -actor:
            break
        line.append(board[y + i][x])
    lines.append(line)

    half_dis = 4 + 1

    line = []  # \
    for i in range(1, half_dis):
        if x - i < 0 or y - i < 0 or board[y - i][x - i] == -actor:
            break
        line.insert(0, board[y - i][x - i])

    for i in range(0, half_dis):
        if x + i == w or y + i == h or board[y + i][x + i] == -actor:
            break
        line.append(board[y + i][x + i])
    lines.append(line)

    line = []  # /
    for i in range(1, half_dis):
        if x - i < 0 or y + i == h or board[y + i][x - i] == -actor:
            break
        line.insert(0, board[y + i][x - i])

    for i in range(0, half_dis):
        if x + i == w or y - i < 0 or board[y - i][x + i] == -actor:
            break
        line.append(board[y - i][x + i])
    lines.append(line)
    return lines


def be5(line, actor):
    c = 0
    for n in line:
        if n == actor:
            c += 1
            if c == 5:
                return True
        else:
            c = 0
    return False


def evaluate(board, x, y, actor):
    lines = get4lines(board, x, y, actor)
    sum = 0
    for l in lines:
        if be5(l, actor):
            print(l, Pattern.FIVE)
            sum += Pattern.FIVE.value
            continue
        nl = l.copy()
        while len(nl) > 6:
            if nl[-2:] == [0, 0]:
                nl.pop()
            elif nl[:2] == [0, 0]:
                nl.pop(0)
            else:
                break
        while len(nl) > 6:
            if nl[-1] == 1:
                nl.pop()
            elif nl[0] == 1:
                nl.pop(0)
        if len(nl) > 5 and nl[0] == 1 and nl[-1] == 0:
            nl.pop()
        elif len(nl) > 5 and nl[0] == 0 and nl[-1] == 1:
            nl.pop(0)

        v = pattern_map.get(tuple(nl))
        print(l, nl, v)
        if v is not None:
            sum += v.value
    return sum


if __name__ == "__main__":
    import numpy as np

    board = np.zeros((10, 10), dtype=np.int32)

    board[0][4] = 1
    board[1][2] = 1
    board[1][3] = 1
    board[1][4] = 1
    board[1][5] = 1
    # board[1][6] = 1
    board[1][7] = 1
    board[1][8] = 1
    # board[1][6] = -1
    v = evaluate(board, 4, 1, 1)
    print("V:", v)
    v = evaluate(board, 4, 0, 1)
    print("V:", v)
