import random
import numpy as np
import time

dictLarge = {}
dictSmall = {}
numbers = np.arange(65536)
random.shuffle(numbers)
for n in numbers:
    dictLarge[n] = random.randint(0, 1024)
    # print(n)

numbers = np.arange(2036)
random.shuffle(numbers)
for n in numbers:
    dictSmall[n] = random.randint(0, 1024)
    # print(n)

st = time.time()
for i in range(0, 10000000):
    dictLarge[random.randint(0, 2035)]
print(time.time() - st)

st = time.time()
for i in range(0, 10000000):
    dictSmall[random.randint(0, 2035)]
print(time.time() - st)




