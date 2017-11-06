import numpy as np
lines = np.load("nursery.txt",delimiter=',')
# lines = open('nursery.txt', 'r').read()
print lines[1]
