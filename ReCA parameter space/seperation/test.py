import numpy as np
import matplotlib.pyplot as plt

# Define the functions to compare
def func1(x):
    return np.sin(x)

def func2(x):
    return np.sqrt(x)

def func3(x):
    return np.exp(-x)

# Define the x range and step size
x = np.arange(0, 5, 0.1)

# Plot the functions and their limits from below
plt.plot(x, func1(x), label='sin(x)')
plt.plot(x, func2(x), label='sqrt(x)')
plt.plot(x, func3(x), label='exp(-x)')

plt.plot(x, np.zeros_like(x), 'k--')

plt.axhline(y=0, color='k', linestyle='--')

plt.xlim(0, 5)
plt.ylim(-1.5, 2)

plt.legend()

plt.show()
