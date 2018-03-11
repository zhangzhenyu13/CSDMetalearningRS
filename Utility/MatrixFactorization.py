import numpy as np
import json,os,time
from sklearn.decomposition import NMF
def loadGraph(gfile):
    f = open("../data/UserGraph/initGraph/"+gfile,"r")
    graphFile=json.load(f)
    size = graphFile["size"]
    graphMatrix = graphFile["data"]
    f.close()
    print("size=%d"%size)
    resultMatrix = np.array(graphMatrix)
    return resultMatrix,size

def matrix_factorization_demo(R, P, Q, K, steps=500000, alpha=0.0002, beta=0.02):

    for step in range(steps):
        t0=time.time()
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] != 0.0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        m,n=R.shape
        for i in range(m):
            for j in range(n):
                if R[i,j]!=0:
                    e+=pow(R[i,j]-eR[i,j],2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if (step+1)%1000==0:

            print("step %d,e=%f, time=%ds"%(step+1,e,time.time()-t0))
        if e < 0.001:
            break
    return P, Q
def matrix_factorization(R,K):
    M,N=R.shape
    P=np.random.rand(M,K)
    Q=np.random.rand(K,N)
    model=NMF(n_components=2, init='random', random_state=0)
    model.fit_transform(R)
    return P,Q

if __name__ == '__main__':

    R = [[5, 3, 0, 1,8],
        [4, 0, 0, 1,0],
        [1, -1, 0, 5,0],
        [1, 0, 0, -4,5],
        [0, 1, -5, 4,1.4]]

    R = np.array(R,dtype=np.float32)
    R=-R

    N = len(R)
    M = len(R[0])
    K = 3

    P = np.random.rand(N, K)
    Q = np.random.rand(K,M)

    nP, nQ = matrix_factorization_demo(R, P, Q, K)
    nR = np.dot(nP, nQ)

    print(R)
    print(nR)
    print(R-nR)
