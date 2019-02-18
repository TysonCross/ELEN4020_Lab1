import numpy as np

def rank2TensorAdd(A, B, N):
    assert np.shape(A)[0] == np.shape(A)[1], "A not square"
    assert np.shape(B)[0] == np.shape(B)[1], "A not square"
    assert np.shape(A) == np.shape(B), "Tensors not the same shape"
    

    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            C[i][j] = A[i][j] + B[i][j]
       
    return C

def rank2TensorMult(A, B, N):
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C

def rank3TensorAdd(A, B, N):
    C = np.zeros((N,N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j][k] = A[i][j][k] + B[i][j][k]
       
    return C

# def rank3TensorMult(A, B, N):
#     C = np.zeros((N,N,N), dtype=int)
#     for i in range(N):
#         for j in range(N):
#             test = rank2TensorMult(A[:,i,:], B[:,:,j], N)
#             print(test)

def rank3TensorMult(A,B,N):
    C = np.zeros((N,N), dtype=int)
    for x in range(N):
        C = C + rank2TensorMult(A[:,x,:], B[:,:,x], N)   

    return C

def main():
    # N input:
    N = int(input("Enter tensor size: "))

    # -------------------------
    #      Rank 2 Tensors
    # -------------------------
    print("\nRank 2", "\n", "-"*30)

    # Creating tensors A and B (N x N)
    a1 = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
    print("A: \n", a1, "\n")

    b1 = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
    print("B: \n", b1, "\n")

    # 2D Tensor Addition
    c1_add = rank2TensorAdd(a1,b1,N)
    print("C (addition): \n", c1_add, "\n")

    # 2D Tensor Multiplication
    c1_mult = rank2TensorMult(a1,b1,N)
    print("C (multiplication): \n", c1_mult, "\n")

    # ------------------------
    #      Rank 3 Tensors
    # ------------------------
    print("\nRank 3", "\n", "-"*30)
    
    # Creating tensors A and B (N x N)
    a2 = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
    print("A: \n", a2, "\n")

    b2 = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
    print("B: \n", b2, "\n")

    # 3D Tensor Addition
    c2_add = rank3TensorAdd(a2,b2,N)
    print("C (addition): \n", c2_add, "\n")

    # 3D Tensor Multiplication (Contraction)
    c2_mult = rank3TensorMult(a2,b2,N)
    print("C (contraction): \n", c2_mult, "\n")


if __name__ == "__main__":
    main()