import numpy as np

def rowExchange(cB, i, j):
    B = cB.copy()
    B[i-1,:] = cB[j-1,:]
    B[j-1,:] = cB[i-1,:]
    return B

def rowMultiply(cB, i):
    B = cB.copy()
    B[i-1,:] = cB[i-1,:]*(-1)
    return B

def rowAdd(cB, i, j, q):
    B = cB.copy()
    B[i-1,:] = cB[i-1,:]+cB[j-1,:]*q
    return B

def columnExchange(cB, i, j):
    B = cB.copy()
    B[:,i-1] = cB[:,j-1]
    B[:,j-1] = cB[:,i-1]
    return B

def columnMultiply(cB, i):
    B = cB.copy()
    B[:,i-1] = cB[:,i-1]*(-1)
    return B

def columnAdd(cB, i, j, q):
    B = cB.copy()
    B[:,i-1] = cB[:,i-1]+cB[:,j-1]*q
    return B

def rowExchangeOperation(B, Q, Qb, i, j):
    B = rowExchange(B, i, j)
    Qb = rowExchange(Qb, i, j)
    Q = columnExchange(Q, i, j)
    return B, Q, Qb

def rowMultiplyOperation(B, Q, Qb, i):
    B = rowMultiply(B, i)
    Qb = rowMultiply(Qb, i)
    Q = columnMultiply(Q, i)
    return B, Q, Qb

def rowAddOperation(B, Q, Qb, i, j, q):
    B = rowAdd(B, i, j, q)
    Qb = rowAdd(Qb, i, j, q)
    Q = columnAdd(Q, i, j, -q)
    return B, Q, Qb

def columnExchangeOperation(B, R, Rb, i, j):
    B = columnExchange(B, i, j)
    R = columnExchange(R, i, j)
    Rb = rowExchange(Rb, i, j)
    return B, R, Rb

def columnMultiplyOperation(B, R, Rb, i):
    B = columnMultiply(B, i)
    R = columnMultiply(R, i)
    Rb = rowMultiply(Rb, i)
    return B, R, Rb

def columnAddOperation(B, R, Rb, i, j, q):
    B = columnAdd(B, i, j, q)
    R = columnAdd(R, i, j, q)
    Rb = rowAdd(Rb, i, j, -q)
    return B, R, Rb

def partRowReduce(B, Q, Qb, k, l):
    m = B.shape[0]
    for i in range(k+1,m+1):
        q = B[i-1,l-1] // B[k-1,l-1]
        B,Q,Qb = rowAddOperation(B,Q,Qb,i,k,-q)
    return B, Q, Qb
        
def partColumnReduce(B, R, Rb, k, l):
    n = B.shape[1]
    for i in range(l+1,n+1):
        q = B[k-1,i-1] // B[k-1,l-1]
        B,R,Rb = columnAddOperation(B,R,Rb,i,l,-q)
    return B, R, Rb

def smallestNonzero(v,k):
    v_p = v[k-1:]
    v_i = np.arange(v.size)
    if all(v_p == 0):
        alpha = 0
        i0 = k
    else:
        alpha = np.min( np.abs( v_p[np.nonzero(v_p)] ) )
        i0 = np.min( np.where( (v_i > k-2) & (np.abs(v) == alpha) ) ) +1
    return alpha,i0
# if branch added in order to avoid taking minimum of zero vector

def rowPrepare(B, Q, Qb, k, l):
    alpha, i = smallestNonzero(B[:,l-1], k)
    B, Q, Qb = rowExchangeOperation(B, Q, Qb, k, i)
    return B, Q, Qb

def rowReduce(B, Q, Qb, k, l):
    while any( B[k:,l-1] != 0):
        B, Q, Qb = rowPrepare(B, Q, Qb, k, l)
        B, Q, Qb = partRowReduce(B, Q, Qb, k, l)
    return B, Q, Qb

def rowEchelon(B):
    m = B.shape[0]
    n = B.shape[1]
    Q  = np.eye(m, dtype=int)
    Qb = np.eye(m, dtype=int)
    k = 0
    l = 1
    while k < m:
        while l<=n:
            if any(B[k:,l-1] != 0):
                break
            l = l + 1
        if l == n + 1:
            break
        k = k + 1
        B, Q, Qb = rowReduce(B, Q, Qb, k, l)
    return B, Q, Qb, k

def kernelImage(A):
    B, P, Pb, k = rowEchelon(A.T)
    if k == 0:
        im = np.zeros((A.shape[0],1),dtype=int)
    else:
        im = B.T[:,:k]
    if k == A.shape[1]:
        ker = np.zeros((A.shape[1],1),dtype=int)
    else:
        ker = Pb.T[:,k:]
    return ker,im
#The textbook has an error. The first output should be transpose of P-bar.
#exceptions for A=0 or A=I added.

def minNonzero(B, k):
    v = np.zeros(B.shape[0], dtype=int)
    q = np.zeros(B.shape[0], dtype=int)
    for i in range(1, k):
        v[i-1] = 0
        q[i-1] = 0
    for i in range(k, B.shape[0]+1):
        v[i-1], q[i-1] = smallestNonzero(B[i-1,:], k)
    alpha, i0 = smallestNonzero(v,k)
    return alpha, i0, q[i0-1]

def moveMinNonzero(B, Q, Qb, R, Rb, k):
    alpha, i, j = minNonzero(B,k)
    B, Q, Qb = rowExchangeOperation(B, Q, Qb, k, i)
    B, R, Rb = columnExchangeOperation(B, R, Rb, k, j)
    return B, Q, Qb, R, Rb

def checkForDivisibility(B, k):
    m = B.shape[0]
    n = B.shape[1]
    for i in range(k+1, m+1):
        for j in range(k+1, n+1):
            q = np.floor(B[i-1,j-1]/B[k-1,k-1]).astype(int)
            if q*B[k-1,k-1] != B[i-1,j-1]:
                return False, i, j, q
    return True, 0, 0, 0

def partSmithForm(B, Q, Qb, R, Rb, k):
    m = B.shape[0]
    n = B.shape[1]
    divisible = False
    while(not divisible):
        B, Q, Qb, R, Rb = moveMinNonzero(B, Q, Qb, R, Rb, k)
        B, Q, Qb = partRowReduce(B, Q, Qb, k, k)
        if any(B[k:,k-1]):
            continue
        B, R, Rb = partColumnReduce(B, R, Rb, k, k)
        if any(B[k-1,k:]):
            continue
        divisible, i, j, q = checkForDivisibility(B, k)
        if not divisible:
            B, Q, Qb = rowAddOperation(B, Q, Qb, i, k, 1)
            B, R, Rb = columnAddOperation(B, R, Rb, j, k, -q)
    return B, Q, Qb, R, Rb
#The textbook has an error. The input indices of col.Add.Op. should be j,k, not k,j.

def smithForm(B):
    m = B.shape[0]
    n = B.shape[1]
    Q = np.eye(m, dtype=int)
    Qb = np.eye(m, dtype=int)
    R = np.eye(n, dtype=int)
    Rb = np.eye(n, dtype=int)
    s = 0
    t = 0
    while np.any(B[t:,t:]):
        t += 1
        B, Q, Qb, R, Rb = partSmithForm(B, Q, Qb, R, Rb, t)
        if B[t-1,t-1] < 0:
            B, Q, Qb = rowMultiplyOperation(B, Q, Qb, t)
        if B[t-1,t-1] == 1:
            s += 1
    return B, Q, Qb, R, Rb, s, t

def Solve(A, b): # input A: 2d array, b: 1d array
    B, Q, Qb, R, Rb, s, t = smithForm(A)
    u = np.zeros(A.shape[1], dtype=int)
    c = Qb @ b # python interprits b as a column vector
    Bdg = np.diag(B)[:t]
    q = np.floor(c[:t]/Bdg).astype(int)
    if all(q*Bdg == c[:t]):
        u[:t] = q
    else:
        return "Failue"
    if any(c[t:]):
        return "Failue"
    return R @ u

def quotientGroup(W, V):
    n = V.shape[1]
    m = W.shape[1]
    A = np.zeros((m,n), dtype=int)
    for i in range(1,n+1):
        A[:,i-1] = Solve(W, V[:,i-1]) #input: 2-d array & 1-d array
    B, Q, Qb, R, Rb, s, t = smithForm(A)
    U = W @ Q
    return [U, B, s, t]

def homologyGroupOfChainComplex(D): #input: list of nparray, index starts from 0
    W = []
    V = []
    H = []
    for k in range(0, len(D)):
        W.append([])
        V.append([])
        H.append([])
    for k in range(0, len(D)):
        W[k], V[k-1] = kernelImage(D[k]) #V[l] is first stored
    V[-1] = np.zeros((D[-1].shape[1],1),dtype=int)
    for k in range(0, len(D)):
        H[k] = quotientGroup(W[k], V[k])
    return H # output is list of list

## example of definitions

# interval(tuple)
# = (0,1)

# cube(tuple of interval)
# = (
#    (0,1),(3,4)
#   )

# cubicalSet(list of cube)
# = [
#    ((0,1),(2,3)),
#    ((0,0),(2,3))
#   ]

# chain(dictionary of cube)
# = {
#    ((0,1),(3,4)) : 3,
#    ((0,0),(3,4)) : 5,
#    ((0,1),(3,3)) : -1,
#    ((0,1),(4,4)) : 1,
#   }

def canonicalCoordinates(c, K): #input: chain c & cubical set K
    #index of cube list K is 0 to len(K)-1, key of dict c is values of K
    v = np.zeros(len(K),dtype=int) # indeces of v: 0 to nc-1
    for i in range(len(K)):
        if (K[i] in c):
            v[i] = c[K[i]]
    return v

def chainFromCanonicalCoordinates(v, K): #input: coordinate vector v & cubical set K
    c = {}
    for i in range(len(K)):
        if (v[i] != 0):
            c[K[i]] = v[i]
    return c

def primaryFace(Q): #input: cube Q
    L = []
    for i in range(len(Q)):
        if Q[i][0] != Q[i][1]: #endpoints are assigned 2nd dimension of cube list
            Qtmp = list(Q)
            ql = Q[i][0]
            qr = Q[i][1]
            Qtmp[i] = (ql,ql)
            L.append(tuple(Qtmp))
            Qtmp[i] = (qr,qr)
            L.append(tuple(Qtmp))
    return L

def dim(Q): #input: cube Q
    d = 0
    for interval in Q:
        if interval[0] != interval[1]:
           d += 1
    return d

def cubicalChainGroups(K):
# setting E
    E = []
    dmax = 0
    for cube in K:
        d = dim(cube)
        if d > dmax:
            dmax = d
    for i in range(dmax+2):
        E.append([]) #list of empty list of index from 0 to dmax+1
# decomposing K to E. primaryFace of 0-d cubes are sent to E[-1]
    while (K != []):
        Q = K[0]
        K = K[1:]
        k = dim(Q)
        L = primaryFace(Q)
        K = list(set().union(K,L))
        E[k-1] = list(set().union(E[k-1],L))
        E[k] = list (set().union(E[k],(Q,))) # (Q,) is required for union of single cube
    return E[:-1]

def boundaryOperator(Q):
    sgn = 1
    c = {}
    for i in range(len(Q)):
        if Q[i][0] != Q[i][1]:
            Qtmp = list(Q)
            ql = Q[i][0]
            qr = Q[i][1]
            Qtmp[i] = (ql,ql)
            c[tuple(Qtmp)] = -sgn
            Qtmp[i] = (qr,qr)
            c[tuple(Qtmp)] = sgn
            sgn = -sgn
    return c

def boundaryOperatorMatrix(E): # E: cubical chain complex
    D = []
    D.append(np.zeros((1,len(E[0])),dtype=int)) # D_0 is zero row vector
    for k in range(1,len(E)):
        Dk_tmp = np.zeros((len(E[k-1]),len(E[k])),dtype=int)
        for j in range(len(E[k])):
            c = boundaryOperator(E[k][j])
            Dk_tmp[:,j] = np.array(canonicalCoordinates(c,E[k-1]))
        D.append(Dk_tmp)
    return D

def generatorsOfHomology(H, E):
    HG = []
    for k in range(0,len(H)):
        HG.append({'generators':[],'orders':[]})
        U,B,s,t = H[k]
        m = U.shape[1]
        if (t == 0):
            # when ker is empty, quotient group(U,B) is represented as 0 array and s=t=0
            break
        for j in range(s+1,m+1): # the textbook says (s+1,len(E[k])...
            if (j <= t):
                order = B[j-1,j-1]
            else:
                order = 'infinity'
            c = chainFromCanonicalCoordinates(U[:,j-1],E[k])
            HG[k]['generators'].append(c)
            HG[k]['orders'].append(order)
    return HG

def homology(K):
    E = cubicalChainGroups(K)
    D = boundaryOperatorMatrix(E)
    H = homologyGroupOfChainComplex(D)
    HG = generatorsOfHomology(H,E)
    return HG

def preBoundary(z, X): # z:chain, X:cubical set
    if (z == {}):
        return {}
    k = dim(z)
    E = cubicalChainGroups(X)
    D = boundaryOperatorMatrix(E)
    y = canonicalCoordinates(z,E[k])
    x = Solve(D,y)
    return chainFromCanonicalCoordinates(x,E[k+1])
