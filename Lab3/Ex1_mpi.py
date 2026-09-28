from mpi4py import MPI
import random
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# N pode ser passado por linha de comando: mpirun -n 4 python3 exercicio1_mpi.py 600
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

A = None
B = None

#
# 1. Geração das matrizes (apenas no rank 0)
#
if rank == 0:
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]

# Sincroniza todos os processos antes de começar a cronometrar
comm.Barrier()
if rank == 0:
    inicio = time.time()

#
# 2. Distribuição das matrizes com Broadcast
#
A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

#
# 3. Cálculo distribuído: cada processo calcula sua fatia de linhas de C
#
linhas_por_processo = N // size
linha_inicial = rank * linhas_por_processo
# O último processo absorve o resto da divisão, caso N não seja múltiplo de size
linha_final = N if rank == size - 1 else linha_inicial + linhas_por_processo

C_local = []
for i in range(linha_inicial, linha_final):
    linha_c = [0.0] * N
    for j in range(N):
        soma = 0.0
        for k in range(N):
            soma += A[i][k] * B[k][j]
        linha_c[j] = soma
    C_local.append(linha_c)

#
# 4. Coleta dos resultados com Gather
#
resultados = comm.gather(C_local, root=0)

#
# 5. Consolidação e tempo total
#
if rank == 0:
    C = []
    for parte in resultados:
        C.extend(parte)

    fim = time.time()
    tempo_ms = (fim - inicio) * 1000

    print(f"Dimensão N = {N}")
    print(f"Número de processos: {size}")
    print(f"Matriz C possui {len(C)} linhas e {len(C[0])} colunas")
    print(f"Tempo distribuído (MPI): {tempo_ms:.2f} ms")