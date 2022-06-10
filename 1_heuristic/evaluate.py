import numpy as np
from enum import Enum
class Pattern(Enum):
  ONE=10
  TWO=100
  THREE=1000
  FOUR=100000
  FIVE=10000000
  BLOCKED_ONE=1
  BLOCKED_TWO=15
  BLOCKED_THREE=150
  BLOCKED_FOUR=15000

# print(Pattern.BLOCKED_FOUR.value)

lived_four = [[0,1,1,1,1,0]]

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

blocked_four = [[1,1,1,1,0], [0,1,1,1,1]]
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


def evaluate(board, x, y, actor):
  h, w = board.shape
  lines = []
  
  line = [] # -
  for i in range(1,5):
    if x-i < 0 or board[y][x-i] == -actor :
      break
    line.insert(0, board[y][x-i])
    
  for i in range(0,5):
    if x+i == w or board[y][x+i] == -actor :
      break
    line.append(board[y][x+i])
  lines.append(line)
  
  
  line = [] # |
  for i in range(1,5):
    if y-i < 0 or board[y-i][x] == -actor :
      break
    line.insert(0, board[y-i][x])
    
  for i in range(0,5):
    if y+i == h or board[y+i][x] == -actor :
      break
    line.append(board[y+i][x])
  lines.append(line)


  line = [] # \
  for i in range(1,5):
    if x-i < 0 or y-i < 0 or board[y-i][x-i] == -actor :
      break
    line.insert(0, board[y-i][x-i])
    
  for i in range(0,5):
    if x+i == w or y+i == h or board[y+i][x+i] == -actor :
      break
    line.append(board[y+i][x+i])
  lines.append(line)


  line = [] # /
  for i in range(1,5):
    if x-i < 0 or y+i == h or board[y+i][x-i] == -actor :
      break
    line.insert(0, board[y-i][x-i])
    
  for i in range(0,5):
    if x+i == w or y-i < 0 or board[y-i][x+i] == -actor :
      break
    line.append(board[y-i][x+i])
  lines.append(line)

  for l in lines:
    print(l)

board = np.zeros((10,10), dtype=np.int32)

board[0][4] = 1
board[1][2] = 1
board[1][3] = 1
board[1][4] = 1
board[1][5] = 1
board[1][6] = -1

evaluate(board, 4, 1, 1)
