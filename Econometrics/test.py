import numpy as np

from Tools import inverse_matrix

row = list(map(int, input().split()))
mat = []
while row:
    mat.append(row)
    row = list(map(int, input().split()))

print(inverse_matrix(np.array(mat)))
