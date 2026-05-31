import numpy as np

test = np.array([i for i in range(10)])
result = np.zeros([2,2])
result[0]=test.mean(),test.std()
print(result[0])