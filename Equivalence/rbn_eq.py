# Elementary CA includes the CA that exclude computation from one or more of the cells.
# this is a short program to find the 3 sets that do this for left central and right nodes.
# this reduces the ECA to even smaller computational universes

import numpy as np
import helpers.helper as helper
import math as math
import itertools
import random
import time

n = 3
complement = False

ttl = int(math.pow(2, n))
perm = int(math.pow(2, ttl))

st = time.time()
permutations = list(itertools.permutations(np.arange(n)))
print(permutations)
swaps = []
i = 0
# https://stackoverflow.com/a/38065863/6638478
for permutation in permutations:
    swaps.append([])
    for cmb in itertools.combinations(permutation, 2):
        if cmb[0] > cmb[1]:
            swaps[i].insert(0, cmb)
    i += 1
print(swaps)
i = 0
# The order of changes might not be correct. This loop checks and fixes the order if necessary.
for permutation in permutations:
    clone = list(permutation)
    # print(clone)
    for swap in swaps[i]:
        clone[swap[0]], clone[swap[1]] = clone[swap[1]], clone[swap[0]]
    # print(list(permutations[0]))
    if clone != list(permutations[0]):
        while clone != list(permutations[0]):
            random.shuffle(swaps[i])
            clone = list(permutation)
            for swap in swaps[i]:
                clone[swap[0]], clone[swap[1]] = clone[swap[1]], clone[swap[0]]

    i += 1
# print(swaps)

found = {}
lookup = {}
for i in range(0, perm):
    if i in found:
        continue
    binary = helper.int_to_binary_string(i, ttl)
    eq = []
    # eq.append(helper.binary_string_to_int(binary))
    for swap_sett in swaps:
        tt = {}
        for j in range(0, ttl):
            tt[j] = binary[j]
        swapped_tt = {}
        for k in range(0, ttl):
            x = helper.int_to_binary_string(k, n)
            for swap in swap_sett:
                x[swap[0]], x[swap[1]] = x[swap[1]], x[swap[0]]
            swapped_tt[helper.binary_string_to_int(x)] = binary[k]
        # print(swapped_tt)
        eq_rule = []
        for k in range(0, ttl):
            eq_rule.append(swapped_tt[k])
        rule = helper.binary_string_to_int(eq_rule)
        eq.append(rule)
        found[rule] = True
    lookup[i] = eq

#
# for key in lookup.keys():
#     print(key, lookup[key])
# if key <= min(lookup[key]):
#     eq = list(dict.fromkeys(lookup[key]))
#     if key in eq:
#         eq.remove(key)
#     eq.sort()
#     print(key, eq)

# Complement
if complement:
    complement_lookup = {}
    for i in range(0, perm):
        binary = helper.int_to_binary_string(i, ttl)
        f_binary = [i ^ 1 for i in binary]
        f_binary.reverse()
        complement = helper.binary_string_to_int([str(i) for i in f_binary])
        complement_lookup[i] = complement

    for key in lookup.keys():
        for value in lookup[key].copy():
            lookup[key].append(complement_lookup[value])

# cleaning the sets of duplicates
for key in lookup.keys():
    # print(key, lookup[key])
    if key <= min(lookup[key]):
        eq = list(dict.fromkeys(lookup[key]))
        if key in eq:
            eq.remove(key)
        eq.sort()
        print(key, eq)

print(time.time() - st)
