import numpy as np

L = 64
D = 32
C = D

a = np.random.rand(L, D)
b = np.random.rand(D, D)

out1 = np.matmul(a, b)
out2 = np.zeros((L, D))

for d in range(D):
    for c in range(C):
        for l in range(L):
            out2[l][d] += a[l][c] * b[c][d]

diff = out2 - out1
print(np.average(diff))
