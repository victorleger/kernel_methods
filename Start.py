import numpy as np
import pandas as pd
import cvxopt as cp
import os
current_path = os.getcwd()

k = 7
y_pred = []

for l in range(3):
    X = pd.read_csv(os.path.join(current_path, 'Xtr{}.csv'.format(l)))
    n = len(X['seq'])
    d = len(X['seq'][0])
    print(n)
    print(d)
    Xtr = []
    for j in range(n):
        Xtr.append([])
        for i in range(d-k):
            S = indice(X['seq'][j][i:i+k],k)
            Xtr[j].append(S)
    Xtr = np.array(Xtr)

    X = pd.read_csv(os.path.join(current_path, 'Xte{}.csv'.format(l)))
    Xte = []
    for j in range(len(X['seq'])):
        Xte.append([])
        for i in range(d-k):
            S = indice(X['seq'][j][i:i+k],k)
            Xte[j].append(S)
    Xte = np.array(Xte)

    K = np.zeros((n,n))
    for j1 in range(n):
        if j1%100==0:
            print(j1)
        Phi_1 = np.zeros(4**k)
        for i in range(d-k):
            Phi_1[Xtr[j1,i]] += 1
        for j2 in range(n):
            Phi_2 = np.zeros(4**k)
            for i in range(d-k):
                Phi_2[Xtr[j2,i]] += 1
            K[j1,j2] = np.sum(Phi_1*Phi_2)

    y = np.array(pd.read_csv(os.path.join(current_path, 'Ytr{}.csv'.format(l)))['Bound'])
    y = 2*y-1

    C = 1
    lbd = 1e-6
    G = np.zeros((2*n,n))
    G[0:n] = np.diag(y)
    G[n::] = np.diag(-y)
    H = np.zeros(2*n)
    H[0:n] = C*np.ones(n)
    P = cp.matrix(K+lbd*np.eye(n))
    q = cp.matrix(-y, tc='d')
    G = cp.matrix(G, tc='d')
    h = cp.matrix(H, tc='d')

    sol = cp.solvers.qp(P,q,G,h)
    alpha = np.reshape(np.array(sol['x']),(n,))

    alpha_sparse = []
    Xtr_sparse = []
    for i in range(n):
        if np.abs(alpha[i])>1e-6:
            alpha_sparse.append(alpha[i])
            Xtr_sparse.append(Xtr[i])
    Xtr_sparse = np.array(Xtr_sparse)
    alpha_sparse = np.array(alpha_sparse)

    W = np.zeros(4**k)
    for j in range(len(alpha_sparse)):
        Phi = np.zeros(4**k)
        for i in range(d-k):
            Phi[Xtr_sparse[j,i]] += 1
        W += alpha_sparse[j]*Phi

    def f_chap(x):
        Phi = np.zeros(4**k)
        for i in range(len(x)-k):
            Phi[x[i]] += 1
        return np.dot(W,Phi.T)

    for i in range(len(Xte)):
        y_pred.append(f_chap(Xte[i])>0)

y_pred = np.array(y_pred,dtype=int)
df = pd.DataFrame({'Id': np.arange(3000), 'Bound': y_pred})
df.to_csv(os.path.join(current_path,'Yte.csv'),index=False)