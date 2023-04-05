import numpy as np
from scipy.sparse import linalg,coo_matrix,csr_matrix
from time import perf_counter

J = 1

def diag(i,n): #対角項の作成
    a = "{:0>{}b}".format(i, n) #n桁の2進数の作成
    count = int(a[0])^int(a[-1]) #左右でスピンが反転している箇所の個数
    if n == 2:
        return (count-1/2)*(-1/2)
    else:
        for j in range(n-1):
            count += int(a[j])^int(a[j+1])
        return (n-2*count)*J/4

def offdiag_arg(i,n):
    row = []
    col = []
    a = format(i,'b')
    l = len(a)
    for j in range(l-1): #10を01と入れ替える
        if a[j] == "1" and a[j+1] == "0":
            row += [i-2**(l-j-1)+2**(l-(j+1)-1),i]
            col += [i,i-2**(l-j-1)+2**(l-(j+1)-1)]
    if l == n and a[-1] == "0": #先頭の1と末尾の0を入れ替える
        row += [i-2**(l-1)+1,i]
        col += [i,i-2**(l-1)+1]
    return row, col

def hamiltonian(n):
    H = [[[],[],[]] for i in range(n+1)] #lists for each num of |-> spins
    for i in range(2**n):
        a = "{:0>{}b}".format(i, n) #make binary (length n)
        l = len(a)
        a_array = np.array([int(a[i]) for i in range(l)])
        Sz = int(np.dot(a_array,np.ones(l))) # num of |-> spins
        data = [diag(i,n)] #diagonal element
        row = [i]
        col = [i]
        r,c = offdiag_arg(i,n) #offdiagonal element
        l = len(r)
        data += [J/2 for _ in range(l)]
        row += r
        col += c
        H[Sz][0] += data
        H[Sz][1] += row
        H[Sz][2] += col
    return H

def ground(H,n): #energy of ground state
    eig_gr = H[0][0]
    for i in range(1,n+1):
        data,row,col = H[i][0],H[i][1],H[i][2]
        H_coo = coo_matrix((data,(row,col)),shape=(2**n,2**n))
        eig,vec = linalg.eigs(H_coo, k=1)
        if eig < eig_gr:
            eig_gr = eig
    return eig_gr

def calc_time(n):
    rt = []
    for i in range(10):
        start = perf_counter()
        ground(hamiltonian(n),n)
        end = perf_counter()
        rt.append(end-start)
    return (np.mean(rt),np.std(rt))

print(calc_time(10))

#print(hamiltonian(4),4)

