import sys
import numpy as np

file1 = open(sys.argv[1], "r")
count = 0

state = 1
w = 15
state_arr = np.zeros((w, w), dtype=np.uint8)
while True:
    count += 1
    # Get next line from file
    line = file1.readline().strip()
    if line.find(" [[") > -1:
        if state != 1:
            raise Exception("error1 line:" + line)
        state = 2
        line_n = 0

    if state != 2:
        continue

    items = line.partition(" [[")
    if items[1] == "":
        items = line.partition(" [")
    if items[1] == "":
        raise Exception("error2 line:" + line + ":" + str(count))

    line_data = items[2].removesuffix("]").removesuffix("]")

    items = [int(x) for x in line_data.split(" ") if x != ""]
    state_arr[line_n] = items
    # print(items)
    if line_n == w - 1:
        # sys.exit()
        state = 1
        print(state_arr, '\n')
        # state_arr.fill(0)
    else:
        line_n += 1
