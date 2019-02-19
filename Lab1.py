import numpy as np

# def are_all_dimensions_equal_length(A):
#     assert ((len(np.shape(A)) > 1) and (len(np.shape(A)) < 4) ), "Tensors must be of rank 2 or 3"

#     for d in range(len(np.shape(A))-2):
#         print(d)
#         if np.shape(A)[d] != np.shape(A)[d+1]:
#             print("Tensor A is not square")
#             return False

#     return True

# def 

def rank2TensorAdd(A, B, N):
    # Assumption is that both input tensors must be square, and of same rank
    assert np.shape(A)[0] == np.shape(A)[1], "A not square"
    assert np.shape(B)[0] == np.shape(B)[1], "B not square"
    assert np.shape(A) == np.shape(B), "Tensors not the same shape"
    
    C = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            C[i][j] = A[i][j] + B[i][j]
    return C

def rank2TensorMult(A, B, N): 
    # Analogous to matrix multiplication
    # Assumption is that both input tensors must be square, and of same rank
    assert np.shape(A)[0] == np.shape(A)[1], "A not square"
    assert np.shape(B)[0] == np.shape(B)[1], "B not square"
    assert np.shape(A) == np.shape(B), "Tensors not the same rank"
    
    C = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

def rank3TensorAdd(A, B, N):
    # Assumption is that both input tensors must be cubic, and of same rank
    assert (np.shape(A)[0] == np.shape(A)[1] and np.shape(A)[1] == np.shape(A)[2]), "A not a cube"
    assert (np.shape(B)[0] == np.shape(B)[1] and np.shape(B)[1] == np.shape(B)[2]), "B not a cube"
    assert np.shape(A) == np.shape(B), "Tensors not the same rank"

    C = np.zeros((N,N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j][k] = A[i][j][k] + B[i][j][k]
    return C

def rank3TensorMult(A, B, N):
    assert ((np.shape(A)[0] == np.shape(A)[1]) and (np.shape(A)[1] == np.shape(A)[2])), "A not a cube"
    assert ((np.shape(B)[0] == np.shape(B)[1]) and (np.shape(B)[1] == np.shape(B)[2])), "B not a cube"
    assert np.shape(A) == np.shape(B), "Tensors not the same shape"

    C = np.zeros((N,N,N), dtype=int)

    # for x in range(N):
    #     C += rank2TensorMult(A[:,x,:], B[:,:,x], N)

    # for x in range(N):
    #     C += rank2TensorMult(A[x,:,:], B[:,x,:], N)

    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             for l in range(N):
    #                 for d in range(N):
    #                     C[i][j][k][l] += A[d][j][k] * B[i][d][l]
    
    # for x in range(N):
    #     A_slice = np.zeros((N,N), dtype=int)
    #     B_slice = np.zeros((N,N), dtype=int)
    #     A_slice = A[x,:,:]
    #     B_slice = B[:,x,:]
    #     C[i][j][x] += rank2TensorMult(A_slice, B_slice, N)

    for x in range(N):
        C[:,:,x] = rank2TensorMult(A[x,:,:], B[:,x,:], N)
        print("slice[",x,"]:\n",C[:,:,x],"\n", sep="")
    
    # assert(C.all() == C_check.all()), "Methods produce different Tensors"

    return C

def main():
    full_output_on = False

    for N in [2,3]:
        print("-"*60)
        print("N =",N)

        # Rank 2 Tensors
        # Randomly generating tensors A and B (N x N)
        a_2d = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
        b_2d = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)

        # Rank 3 Tensors
        # Randomly generating tensors A and B (N x N x N)
        a_3d = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
        b_3d = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)

        # 2D Tensor Addition
        c_2d_add = rank2TensorAdd(a_2d,b_2d,N)
        
        # 2D Tensor Multiplication
        c_2d_mult = rank2TensorMult(a_2d,b_2d,N)
        
        # 3D Tensor Addition
        c_3d_add = rank3TensorAdd(a_3d,b_3d,N)

        # 3D Tensor Multiplication (Contraction)
        c_3d_mult = rank3TensorMult(a_3d,b_3d,N)

        # Output
        # Rank 2 Tensors
        # print("\n", "-"*5, "Rank 2", "-"*5)
        
        # if full_output_on:
        #     print("\nA:")
        #     print(a_2d)

        #     print("\nB:")
        #     print(b_2d)

        #     print("\nC: (element-wise addition)")
        #     print(c_2d_add)

        # print("\nC: (matrix multiplication)")
        # print(c_2d_mult)

        # Rank 3 Tensors
        print("\n", "-"*5, "Rank 3", "-"*5)

        if full_output_on:
            print("\nA:")
            print(a_3d)

            print("\nB:")
            print(b_3d)

            print("\nC: (element-wise addition)")
            print(c_3d_add)

        print("\nC: (tensor contraction)")
        print(c_3d_mult)


if __name__ == "__main__":
    main()