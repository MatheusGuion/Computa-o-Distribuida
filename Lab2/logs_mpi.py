from mpi4py import MPI
import random

# Inicialização do MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#
# Função para gerar logs
#
def gerar_logs(qtd):
    ips = [f"192.168.1.{i}" for i in range(2, 254)]
    endpoints = [
        "/",
        "/login",
        "/products",
        "/cart",
        "/checkout",
        "/api/users",
        "/api/orders"
    ]
    metodos = ["GET", "POST"]
    status = ["200", "200", "200", "404", "500"]

    logs = []
    for _ in range(qtd):
        ip = random.choice(ips)
        metodo = random.choice(metodos)
        endpoint = random.choice(endpoints)
        cod_status = random.choice(status)
        logs.append(f"{ip} {metodo} {endpoint} {cod_status}")

    return logs


#
# Processo 0 gera o dataset
#
logs_divididos = None
TOTAL_LOGS = 100000

if rank == 0:
    print("\nGerando dataset de logs...\n")
    logs = gerar_logs(TOTAL_LOGS)

    # DIVIDIR AQUI O DATASET ENTRE OS PROCESSOS
    # O NÓ MASTER TAMBÉM PROCESSA SUA PARTE DO DATASET
    tamanho_parte = len(logs) // size
    logs_divididos = [
        logs[i * tamanho_parte: (i + 1) * tamanho_parte]
        for i in range(size)
    ]

#
# Distribuição usando Scatter
#
logs_locais = comm.scatter(logs_divididos, root=0)

#
# Processamento local em cada nó
#
erros = 0
for linha in logs_locais:
    campos = linha.split()
    cod_status = campos[3]
    if cod_status in ("404", "500"):
        erros += 1

mensagem = (
    f"Processo {rank} analisou {len(logs_locais)} linhas "
    f"e encontrou {erros} erros."
)

#
# Nó Master
# Recebe (via mensagens ponto a ponto) e imprime os resultados de cada
# worker, além dos seus próprios.
#
if rank == 0:
    print(mensagem)
    for origem in range(1, size):
        msg_recebida = comm.recv(source=origem, tag=0)
        print(msg_recebida)
else:
    comm.send(mensagem, dest=0, tag=0)