import random
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

A = [[random.random() for _ in range(N)] for _ in range(N)]
B = [[random.random() for _ in range(N)] for _ in range(N)]
C = [[0] * N for _ in range(N)]

inicio = time.time()
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
fim = time.time()

print(f"N = {N}")
print(f"Tempo sequencial: {(fim - inicio) * 1000:.2f} ms")