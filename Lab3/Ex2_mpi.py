from mpi4py import MPI
import random
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# N pode ser passado por linha de comando: mpirun -n 4 python3 exercicio2_mpi.py 10000000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

#
# 1. Particionamento: cada processo gera N_local pontos
#
N_local = N // size
# O rank 0 absorve o resto da divisão, caso N não seja múltiplo de size
if rank == 0:
    N_local += N % size

comm.Barrier()
inicio = MPI.Wtime()

#
# 2. Geração e contagem local
#
dentro_local = 0
for _ in range(N_local):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1.0:
        dentro_local += 1

#
# 3. Agregação dos resultados com Reduce
#
dentro_total = comm.reduce(dentro_local, op=MPI.SUM, root=0)

#
# 4. Cálculo final (apenas no processo 0)
#
if rank == 0:
    fim = MPI.Wtime()
    pi_estimado = 4.0 * dentro_total / N
    tempo_ms = (fim - inicio) * 1000

    print(f"N = {N} pontos, {size} processos")
    print(f"PI aproximado: {pi_estimado:.6f}")
    print(f"Tempo distribuído (MPI): {tempo_ms:.2f} ms")