import numpy as np

def rank2TensorAdd(A, B, N):
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

def rank3TensorMult(A, B, N):
    C = np.zeros((N,N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            test = rank2TensorMult(A[:,i,:], B[:,:,j], N)
            print(test)

def rank3TensorMult2(A,B,N):
    C = np.zeros((N,N), dtype=int)
    for x in range(N):
        print(rank2TensorMult(A[:,x,:], B[:,:,x], N))
        C = C + rank2TensorMult(A[:,x,:], B[:,:,x], N)   

    return C

def main():
    # Rank 2 Tensors
    print("Rank 2")
    N = int(input("Enter size: "))

    a1 = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
    print(a1, "\n")
    print("-----------------------------------")

    b1 = np.random.uniform(low=0, high=20, size=(N,N)).astype(int)
    print(b1, "\n")
    print("-----------------------------------")

    c1_add = rank2TensorAdd(a1,b1,N)
    print(c1_add, "\n")
    print("-----------------------------------")

    c1_mult = rank2TensorMult(a1,b1,N)
    print(c1_mult, "\n")
    print("-----------------------------------")

    # Rank 3 Tensors
    print("Rank 3")
    print("-----------------------------------")
    
    a2 = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
    print(a2, "\n")
    print("-----------------------------------")

    b2 = np.random.uniform(low=0, high=20, size=(N,N,N)).astype(int)
    print(b2, "\n")
    print("-----------------------------------")

    # c2_add = rank3TensorAdd(a2,b2,N)
    # print(c2_add, "\n")
    # print("-----------------------------------")

    print(rank3TensorMult2(a2,b2,N), "\n")
    print(np.tensordot(a2,b2), "\n")
    print(np.matmul(a2,b2))


if __name__ == "__main__":
    main()