from enum import Enum
import numpy as np

__all__ = ["evaluate_one", "evaluate_all", "evaluate_4dir_lines", "StonePattern"]


class StonePattern(Enum):
    ONE = 10
    TWO = 100
    THREE = 10000
    FOUR = 100000
    FIVE = 10000000
    BLOCKED_ONE = 1
    BLOCKED_TWO = 15
    BLOCKED_THREE = 150
    BLOCKED_FOUR = 15000

    # print(StonePattern.BLOCKED_FOUR.value)


lived_four = [[0, 1, 1, 1, 1, 0]]

pattern_map = {(1, 1, 1, 1, 1): StonePattern.FIVE}
pattern_search_order = [[(1, 1, 1, 1, 1), 5]]

for four_p in lived_four:
    tmp4 = four_p.copy()
    pattern_map[tuple(tmp4)] = StonePattern.FOUR
    pattern_search_order.append([tuple(tmp4), sum(tmp4)])
    for i3 in range(len(tmp4)):
        if tmp4[i3] == 1:
            tmp3 = tmp4.copy()
            tmp3[i3] = 0
            pattern_map[tuple(tmp3)] = StonePattern.THREE
            pattern_search_order.append([tuple(tmp3), sum(tmp3)])
            for i2 in range(len(tmp3)):
                if tmp3[i2] == 1:
                    tmp2 = tmp3.copy()
                    tmp2[i2] = 0
                    pattern_map[tuple(tmp2)] = StonePattern.TWO
                    pattern_search_order.append([tuple(tmp2), sum(tmp2)])
                    for i1 in range(len(tmp2)):
                        if tmp2[i1] == 1:
                            tmp1 = tmp2.copy()
                            tmp1[i1] = 0
                            pattern_map[tuple(tmp1)] = StonePattern.ONE
                            pattern_search_order.append([tuple(tmp1), sum(tmp1)])

print(pattern_map)

blocked_four = [[1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1]]
for blocked_four_p in blocked_four:
    tmp4 = blocked_four_p.copy()
    pattern_map[tuple(tmp4)] = StonePattern.BLOCKED_FOUR
    pattern_search_order.append([tuple(tmp4), sum(tmp4)])
    for i3 in range(len(tmp4)):
        if tmp4[i3] == 1:
            tmp3 = tmp4.copy()
            tmp3[i3] = 0
            pattern_map[tuple(tmp3)] = StonePattern.BLOCKED_THREE
            pattern_search_order.append([tuple(tmp3), sum(tmp3)])
            for i2 in range(len(tmp3)):
                if tmp3[i2] == 1:
                    tmp2 = tmp3.copy()
                    tmp2[i2] = 0
                    pattern_map[tuple(tmp2)] = StonePattern.BLOCKED_TWO
                    pattern_search_order.append([tuple(tmp2), sum(tmp2)])
                    for i1 in range(len(tmp2)):
                        if tmp2[i1] == 1:
                            tmp1 = tmp2.copy()
                            tmp1[i1] = 0
                            pattern_map[tuple(tmp1)] = StonePattern.BLOCKED_ONE
                            pattern_search_order.append([tuple(tmp1), sum(tmp1)])

pattern_search_order.sort(key=lambda x: pattern_map[x[0]].value, reverse=True)

# for k in pattern_search_order:
#     print(k, pattern_map[k])

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

    if actor == -1:
        for line in lines:
            for i in range(len(line)):
                line[i] = line[i] * -1
    return lines


def be5(line, actor):  # 判断是否 >= 五连
    c = 0
    for n in line:
        if n == actor:
            c += 1
            if c == 5:
                return True
        else:
            c = 0
    return False


def evaluate_one(board, x, y, actor):
    lines = get4lines(board, x, y, actor)
    sum = 0
    for line in lines:
        if be5(line, actor):
            # print(x, y, line, StonePattern.FIVE)
            sum += StonePattern.FIVE.value
            continue
        nl = line.copy()
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
        # print(x, y, line, nl, v)
        if v is not None:
            sum += v.value
    return sum


eva_cache = dict()
def evaluate_1line(iline):
    sum = eva_cache.get(tuple(iline))
    if sum is not None:
        return sum
    # print("G1")
    lines = []
    sum = 0
    if len(iline) < 5:
        return 0

    tmp = []
    for idx in range(len(iline)):
        if iline[idx] != -1:
            tmp.append(iline[idx])
        else:
            if len(tmp) >= 5:
                lines.append(tmp)
            tmp = []
    if len(tmp) >= 5:
        lines.append(tmp)
    # print("G2")
    # sum = 0
    # lines=[iline.tolist()]
    # print("evaluate_1line:", lines[0])
    while len(lines) > 0:
        line = lines.pop()
        while len(line) > 5:
            if line[:5] == [0, 0, 0, 0, 0]:
                line.pop(0)
                # print("pop", line)
            elif line[-5:] == [0, 0, 0, 0, 0]:
                line.pop()
            else:
                break
        # print("G3")
        # line = np.array(line)
        line_l = len(line)
        line_1cnt = line.count(1)
        if line_l < 5 or line_1cnt == 0:
            continue

        found = False
        for pat, pat_1cnt in pattern_search_order:
            if pat_1cnt > line_1cnt:
                continue
            # print("pat", pat)
            pat_l = len(pat)
            for i in range(0, line_l - pat_l + 1):
                # print("pat0", pat, line[i : i + pat_l])
                if line[i : i + pat_l] == list(pat):
                    # print("Got", pat, line[i : i + pat_l])
                    # print("Got P:", line[i : i + pat_l], pat, pattern_map[pat])
                    sum += pattern_map[pat].value
                    if i > 0:
                        lines.append(line[:i])
                    if line_l > i + pat_l:
                        lines.append(line[i + pat_l :])
                    found = True
                    break
            if found:
                break
    eva_cache[tuple(iline)] = sum
    return sum


def evaluate_4dir_lines(board, x, y):
    h, w = board.shape
    nb = board.copy()

    v11 = evaluate_1line(nb[:, x])  # |
    v12 = evaluate_1line(nb[y])  # -
    v13 = evaluate_1line(nb.diagonal(x - y))  # \
    # print("G ", w, w - 1 - (x - y), np.fliplr(nb).diagonal(w - 1 - (x + y)))
    # print(nb)
    # print(np.fliplr(nb))
    v14 = evaluate_1line(np.fliplr(nb).diagonal(w - 1 - (x + y)))  # /

    nb *= -1
    v21 = evaluate_1line(nb[:, x])  # |
    v22 = evaluate_1line(nb[y])  # -
    v23 = evaluate_1line(nb.diagonal(x - y))  # \
    v24 = evaluate_1line(np.fliplr(nb).diagonal(w - 1 - (x + y)))  # /

    return [v11, v21], [v12, v22], [v13, v23], [v14, v24]


def evaluate_all(board, board_score_v, board_score_h, board_score_lu2rb, board_score_lb2ru):
    h, w = board.shape
    nb = board.copy()


if __name__ == "__main__":
    import time

    board = np.zeros((15, 15), dtype=np.int32)

    board[0][4] = 1
    board[1][2] = 1
    board[1][3] = 1
    board[1][4] = 1
    board[1][5] = 1
    # board[1][6] = 1
    board[1][7] = 1
    board[1][8] = 1
    board[14][14] = 1
    # board[1][6] = -1
    v = evaluate_one(board, 4, 1, 1)
    print("V:", v)
    v = evaluate_one(board, 4, 0, 1)
    print("V:", v)
    v = evaluate_one(board, 14, 14, 1)
    print("V:", v)

    # stime = time.time()
    # for epoch in range(100):
    #     for i in range(15):
    #         for j in range(15):
    #             if board[i][j] != 0:
    #                 continue
    #             board[i][j] = 1
    #             evaluate_one(board, j, i, 1)
    #             board[i][j] = 0
    # print("time:", time.time() - stime)

    print("V:", evaluate_4dir_lines(board, 4, 1))
    # print("V:", evaluate_4dir_lines(board, 4, 0, 1))
    # print("V:", evaluate_4dir_lines(board, 14, 14, 1))
    board *= -1
    print("V:", evaluate_4dir_lines(board, 4, 1))



    board = np.zeros((9, 9), dtype=np.int32)

    board[6][3] = 1
    board[5][4] = 1
    board[4][4] = 1
    board[4][5] = 1

    board[6][4] = -1
    board[5][5] = -1
    board[4][6] = -1
    board[3][7] = -1
    print("1V:", evaluate_4dir_lines(board, 3, 7))
    board *= -1
    print("1V:", evaluate_4dir_lines(board, 3, 7))
