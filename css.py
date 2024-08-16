#import ldpc
import numpy as np
from ldpc.mod2 import *
#.inverse

def main():
    # test inverse
    mat=np.array([[1,1,0],[0,1,0],[0,0,1]])
    i_mat=inverse(mat)
    print(i_mat@mat%2)

    # test nullspace
    H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])    
    print(nullspace(H))

    G = np.concatenate((H,H,H,H),axis=1)
    for j in range(100):
      for i in range(10000):
        a=nullspace(G)
        #print(a)
        #break
      print(j)
    
    # test rre
    rre=reduced_row_echelon(H)[0]
    print(rre)

#main()

def make_it_full_rank(A):
    R,r,_,_ = row_echelon(A)
    return R[:r], r
    
def mat_rand(row,col,p=0.5):
    '''Return a radnom binary matrix of dimension (r,c)
    '''
    R = np.random.rand(row,col)
    B = np.floor(R + p).astype(int)
    if True:
        # make it full rank
        F,r = make_it_full_rank(B)
    return F,r

class CSSCode:
    def __init__(self):
        pass

def min_distance(Hx,Lx):
    # return dx
    r,n = Hx.shape
    k = Lx.shape[0]
    alpha = np.zeros((2**k-1,k),int)
    for i in range(1,2**k):
        b=mod10_to_mod2(i,k)
        alpha[i-1]=b
    #print(f"{alpha = }")

    alpha_Lx = alpha @ Lx

    dx = n
    for j in range(0, 2**r):
        b=mod10_to_mod2(j,r)
        beta = b @ Hx
        codewords = np.add(alpha_Lx , beta) %2
        #print(codewords)
        s = codewords.sum(axis=1)
        m = s.min()
        #print(s,m)
        if m < dx:
            dx = m
    return dx
    

def match(A,B):
    # assert A and B are identical
    assert not ( A - B ).any()
    
def css_test():
    #print('get a random CSS code')
    n=10
    rx=3
    rz=5
    Hx,rx = mat_rand(rx,n)
    #print(f"{Hx = }")
    U = nullspace(Hx)
    theta, rz = mat_rand(rz,n-rx)
    Hz = theta@U %2
    #print(f"{Hz = }")

    # check commutation
    Hx_Hz = Hx@np.transpose(Hz) %2
    assert not Hx_Hz.any()

    Ux = nullspace(Hz)
    Uz = U
    a = Ux @ np.transpose(Uz) % 2
    r,k,P,Q = reduced_row_echelon(a)
    match(r, (P @ Ux ) @  (Q.T @ Uz).T %2 )
    
    #print('# of logical qubits k=',k)
    Lx = P[:k] @ Ux %2
    Lz = Q.T[:k] @ Uz %2
    #print(f"{Lx = }")
    #print(f"{Lz = }")

    #check commutation
    Lx_Lz = Lx @ Lz.T %2
    assert not ( Lx_Lz - np.identity(k) ).any()
    assert not (Hx @ Lz.T %2).any()
    assert not (Hz @ Lx.T %2).any()

    #check distance
    dx = min_distance(Hx,Lx)
    dz = min_distance(Hz,Lz)
    print(f"{n = }, {k = }, {dx = }, {dz = }")
    
css_test()

for i in range(1000000):
    print(i)
    css_test()
