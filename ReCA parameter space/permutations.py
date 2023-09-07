

import numpy as np
import itertools

def printRolls(a, x):
    # print(x)
    # print(tuple(x[0]))
    b = np.roll(a, 1)
    c = np.roll(a, 2)
    d = np.roll(a, 3)
    print(f"{a}:{x.index(tuple(a))} {b}:{x.index(tuple(b))} {c}:{x.index(tuple(c))} {d}:{x.index(tuple(d))}")

    # x.remove(tuple(a))
    # x.remove(tuple(b))
    # x.remove(tuple(c))
    # x.remove(tuple(d))
n= 4

x = list(itertools.permutations(np.arange(n)))
print(x)


printRolls(np.array([0,1,2,3]), x)
printRolls(np.array([0,2,1,3]), x)

printRolls(np.array([2,1,0,3]), x)
printRolls(np.array([0,1,3,2]), x)

printRolls(np.array([3,1,0,2]), x)
printRolls(np.array([0,3,1,2]), x)

# states = 3
# n = 6
# x = []
#
#
# def acc(past, depth):
#     for i in range(0, states):
#         current = past.copy()
#         current.append(i)
#         if depth == n:
#             print(current)
#             current.sort()
#             x.append(''.join(map(str, current)))
#         else:
#             acc(current, depth + 1)
#
#
# acc([], 1)
#
# x.sort()
# print(x)
# print(len(x))
#
# print(list(dict.fromkeys(x)))
# print(len(list(dict.fromkeys(x))))
