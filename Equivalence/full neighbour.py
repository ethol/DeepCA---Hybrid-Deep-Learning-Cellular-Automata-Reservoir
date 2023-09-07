import itertools
import numpy as np
import helpers.helper as helper

sizexny = 18

initialization = np.random.randint(1, size=(sizexny, sizexny))

pos_neighbours = []
for i in range(0, 32):
    pos_neighbours.append(list(helper.int_to_binary_string(i, 5)))

    # lookuptable[''.join(str(bin))] = random.randint(0, 1)
count = 0
for x in range(0, sizexny-2, 3):
    for y in range(0, sizexny-2, 3):
        if count < len(pos_neighbours):
            place = pos_neighbours[count]
            initialization[x+1][y] = place[0]
            initialization[x][y+1] = place[1]
            initialization[x+1][y+1] = place[2]
            initialization[x+2][y+1] = place[3]
            initialization[x+1][y+2] = place[4]
        count += 1


np.set_printoptions(threshold=np.sys.maxsize)
print(initialization)
print(pos_neighbours)
