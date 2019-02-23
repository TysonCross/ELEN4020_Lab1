# Tyson Cross       1239448
# Michael Nortje    1389486 
# Josh Isserow      675720

import numpy as np

def checkRank2Valid(A,B):
    assert np.shape(A)[0] == np.shape(A)[1], "A not square"
    assert np.shape(B)[0] == np.shape(B)[1], "B not square"
    assert np.shape(A) == np.shape(B), "Tensors not the same shape"

def checkRank3Valid(A,B):
    assert (np.shape(A)[0] == np.shape(A)[1] and np.shape(A)[1] == np.shape(A)[2]), "A not a cube"
    assert (np.shape(B)[0] == np.shape(B)[1] and np.shape(B)[1] == np.shape(B)[2]), "B not a cube"
    assert np.shape(A) == np.shape(B), "Tensors not the same rank"

def rank2TensorAdd(A, B, N):
    # Element-wise addition
    # Assumption is that both input tensors must be square, and of same rank
    checkRank2Valid(A,B)
    C = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            C[i][j] = A[i][j] + B[i][j]
    return C

def rank2TensorMult(A, B, N): 
    # Analogous to matrix multiplication
    # Assumption is that both input tensors must be square, and of same rank
    checkRank2Valid(A,B)
    C = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

def rank3TensorAdd(A, B, N):
    # Assumption is that both input tensors must be cubic, and of same rank
    checkRank3Valid(A,B)
    C = np.zeros((N,N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j][k] = A[i][j][k] + B[i][j][k]
    return C

def rank3TensorMult(A, B, N):
    # Iterates (with the free index 'x') through a commonly-indexed "slice" of each input rank 3 tensor.
    # This produces two 2d matrices. Matrix multiplication is performed on these two matrices.
    # The resulting 2d matrix is assigned to the C[][][x] slice. 
    # Assumption is that both input tensors must be square, and of same rank
    checkRank3Valid(A,B)
    C = np.zeros((N,N,N), dtype=int)
    for x in range(N):
        C[:,:,x] = rank2TensorMult(A[x,:,:], B[:,x,:], N)
    
    return C

def main():
    for N in [10,20]:
        #------------------
        #   Calculations
        #------------------
        
        # Randomly generating tensors A and B (N x N) (Rank 2)
        a_2d = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
        b_2d = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)

        # Randomly generating tensors A and B (N x N x N) (Rank 3)
        a_3d = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
        b_3d = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)

        # 2D tensor operations
        c_2d_add = rank2TensorAdd(a_2d,b_2d,N)
        c_2d_mult = rank2TensorMult(a_2d,b_2d,N)
        
        # 3D tensor operations
        c_3d_add = rank3TensorAdd(a_3d,b_3d,N)
        c_3d_mult = rank3TensorMult(a_3d,b_3d,N)

        #--------------
        #   Output
        #--------------
        terminal_width = 120
        np.set_printoptions(threshold=np.inf,linewidth=terminal_width)
        title = " N = " + str(N) +": "
        print("\n",title.center(terminal_width,"*"))

        # Rank 2 Tensors output
        print("\n","Rank 2:\n","-"*7)

        print("\nA:\n\n",a_2d)
        print("\nB:\n\n",b_2d)
        print("\nC: (element-wise addition)\n\n",c_2d_add)
        print("\nC: (matrix multiplication)\n\n",c_2d_mult)

        # Rank 3 Tensors output
        print("\n","- "*int(terminal_width/2))
        print("\n","Rank 3:\n","-"*7)

        print("\nA:\n\n",a_3d)
        print("\nB:\n\n",b_3d)
        print("\nC: (element-wise addition)\n\n",c_3d_add)
        print("\nC: (tensor contraction)\n\n",c_3d_mult,"\n")


if __name__ == "__main__":
    main()