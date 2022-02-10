import math
from scipy.spatial import distance
import numpy as np

G = []
X = [[27, 36], [12, 18], [57, 89], [99, 100], [
    178, 124], [134, 111], [145, 167], [11, 14]]
Y = [[45, 59], [26, 43], [67, 54], [88, 90], [
    178, 142], [123, 134], [134, 187], [9, 7]]
N = len(X)

# make G
for i in range(N):
    GT = []
    for j in range(N):
        dist = distance.euclidean(np.array(Y[j]), np.array(X[i]))
        theta = math.log(dist)*(dist**2)
        GT.append(theta)
    G.append(GT)

# make T and Yp
T = np.zeros((11, 11))
Yp = np.zeros((11, 2))

G = np.array(G)
G = np.transpose(G)


for i in range(N):
    for j in range(N):
        T[i][j] = G[i][j]

for i in range(N):
    Yp[i][0] = Y[i][0]
    Yp[i][1] = Y[i][1]

    T[i][8] = T[8][i] = 1
    T[i][9] = T[9][i] = X[i][0]
    T[i][10] = T[10][i] = X[i][1]


W = np.dot(T, Yp)

# results
A = W[8:11]
print("W is:\n", W)
print("where A is:\n", A)
