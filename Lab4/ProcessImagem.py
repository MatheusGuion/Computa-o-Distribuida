from mpi4py import MPI
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parâmetros padrão de teste (podem ser sobrescritos via linha de comando:
# mpirun -n 4 python3 processamento_imagens.py 2000 2000)
LINHAS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
COLUNAS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
LIMIAR_SUSPEITO = 200
LIMIAR_ALTO = 230
PCT_CRITICO = 5.0  # percentual em %

imagem_completa = None
parametros = None

t_inicio = time.time()


def gerar_radiografia(linhas, colunas):
    """Gera uma radiografia sintética com um foco suspeito injetado no pulmão
    direito. A região do foco é proporcional ao tamanho da imagem, para que a
    simulação funcione tanto em imagens pequenas (500x500) quanto grandes
    (6000x6000)."""
    imagem = np.random.randint(30, 90, size=(linhas, colunas), dtype=np.uint8)

    altura_foco = max(1, linhas // 4)
    largura_foco = max(1, colunas // 4)
    linha_inicio_foco = min(linhas // 3, linhas - altura_foco)
    coluna_inicio_foco = min(int(colunas * 0.6), colunas - largura_foco)

    imagem[
        linha_inicio_foco:linha_inicio_foco + altura_foco,
        coluna_inicio_foco:coluna_inicio_foco + largura_foco
    ] = np.random.randint(180, 245, size=(altura_foco, largura_foco), dtype=np.uint8)

    return imagem


#
# Etapa 2 & 3: Processo 0 gera a radiografia e define os parâmetros globais
#
if rank == 0:
    print(f"[Master] Inicializando análise com {size} processos MPI "
          f"(imagem {LINHAS}x{COLUNAS})...")

    imagem_completa = gerar_radiografia(LINHAS, COLUNAS)

    parametros = {
        'linhas': LINHAS,
        'colunas': COLUNAS,
        'limiar_suspeito': LIMIAR_SUSPEITO,
        'limiar_alto': LIMIAR_ALTO,
        'pct_critico': PCT_CRITICO
    }

# Etapa 3: Difusão dos parâmetros via Broadcast
parametros = comm.bcast(parametros, root=0)

# Etapa 4: Sincronização inicial — garante que todos os nós já têm os
# parâmetros antes de iniciar o tráfego pesado de dados da imagem
comm.Barrier()

#
# Etapa 5: Particionamento e distribuição com Scatter
#
# Desafio de divisibilidade: se o número de linhas não for múltiplo de
# 'size', usamos padding — completamos a imagem com linhas de zeros até o
# próximo múltiplo de 'size', dividimos em blocos iguais e, depois, cada
# processo descarta localmente as linhas de padding que recebeu antes de
# calcular suas estatísticas (para não distorcer médias e contagens).
linhas_por_proc = int(np.ceil(parametros['linhas'] / size))

blocos_divididos = None
if rank == 0:
    total_com_padding = linhas_por_proc * size
    padding = total_com_padding - parametros['linhas']
    if padding > 0:
        linhas_padding = np.zeros((padding, parametros['colunas']), dtype=np.uint8)
        imagem_padded = np.vstack([imagem_completa, linhas_padding])
        print(f"[Master] Linhas não divisíveis por {size} processos: "
              f"adicionadas {padding} linha(s) de padding (descartadas na análise).")
    else:
        imagem_padded = imagem_completa
    blocos_divididos = np.split(imagem_padded, size, axis=0)

bloco_local = comm.scatter(blocos_divididos, root=0)

#
# Etapa 6 & 7: Processamento e classificação local
#
linha_inicio_bloco = rank * linhas_por_proc
linha_fim_bloco = linha_inicio_bloco + linhas_por_proc  # exclusivo, no espaço com padding
linhas_reais_no_bloco = max(0, min(linha_fim_bloco, parametros['linhas']) - linha_inicio_bloco)

if linhas_reais_no_bloco > 0:
    bloco_valido = bloco_local[:linhas_reais_no_bloco]

    total_local = bloco_valido.size
    soma_local = int(np.sum(bloco_valido))
    max_local = int(np.max(bloco_valido))

    col_meio = parametros['colunas'] // 2
    esq_mask = bloco_valido[:, :col_meio] > parametros['limiar_suspeito']
    dir_mask = bloco_valido[:, col_meio:] > parametros['limiar_suspeito']
    suspeitos_esq = int(np.sum(esq_mask))
    suspeitos_dir = int(np.sum(dir_mask))
    suspeitos_local = suspeitos_esq + suspeitos_dir
    altamente_suspeitos_local = int(np.sum(bloco_valido > parametros['limiar_alto']))

    pct_suspeito = (suspeitos_local / total_local) * 100.0
    if pct_suspeito >= parametros['pct_critico']:
        classificacao_local = "CRÍTICA"
    elif pct_suspeito >= 1.0:
        classificacao_local = "ATENÇÃO"
    else:
        classificacao_local = "NORMAL"

    linha_fim_real = min(linha_fim_bloco, parametros['linhas']) - 1
else:
    # Bloco composto inteiramente por padding (só ocorre em casos extremos,
    # ex.: mais processos do que linhas de imagem)
    total_local = 0
    soma_local = 0
    max_local = 0
    suspeitos_esq = 0
    suspeitos_dir = 0
    suspeitos_local = 0
    altamente_suspeitos_local = 0
    classificacao_local = "SEM DADOS (padding)"
    linha_fim_real = linha_inicio_bloco - 1

#
# Etapa 8: Simulação de heterogeneidade de hardware
#
if rank % 2 != 0:
    time.sleep(0.5 * rank)

#
# Etapa 9: Sincronização pré-consolidação — evidencia o straggler effect,
# já que todos os processos rápidos ficam esperando o mais lento aqui
#
comm.Barrier()

#
# Etapa 10: Consolidação numérica global com Reduce
#
total_pixels_global = comm.reduce(total_local, op=MPI.SUM, root=0)
soma_global = comm.reduce(soma_local, op=MPI.SUM, root=0)
max_global = comm.reduce(max_local, op=MPI.MAX, root=0)
suspeitos_global = comm.reduce(suspeitos_local, op=MPI.SUM, root=0)
altos_global = comm.reduce(altamente_suspeitos_local, op=MPI.SUM, root=0)
esq_global = comm.reduce(suspeitos_esq, op=MPI.SUM, root=0)
dir_global = comm.reduce(suspeitos_dir, op=MPI.SUM, root=0)

#
# Etapa 11: Coleta de relatórios descritivos individuais com Gather
#
relatorio_local = {
    'rank': rank,
    'linhas_inicio': linha_inicio_bloco,
    'linhas_fim': linha_fim_real,
    'pixels': total_local,
    'suspeitos': suspeitos_local,
    'classificacao': classificacao_local,
    'max_local': max_local
}
todos_relatorios = comm.gather(relatorio_local, root=0)

#
# Etapa 12: Relatório final consolidado e diagnóstico no processo root
#
if rank == 0:
    t_total = (time.time() - t_inicio) * 1000.0
    media_intensidade = soma_global / total_pixels_global
    taxa_comprometida = (suspeitos_global / total_pixels_global) * 100.0

    print("\n" + "=" * 60)
    print(" RELATÓRIO CONSOLIDADO DE TRIAGEM DISTRIBUÍDA")
    print("=" * 60)
    print(f"Dimensões do Exame       : {LINHAS} x {COLUNAS} pixels")
    print(f"Processos MPI Utilizados : {size}")
    print(f"Tempo Total de Execução  : {t_total:.2f} ms")
    print(f"Intensidade Média Global : {media_intensidade:.2f} (Máxima: {max_global})")
    print(f"Total de Pixels Suspeitos: {suspeitos_global} ({taxa_comprometida:.2f}%)")
    print(f"  - Altamente suspeitos  : {altos_global}")
    print(f"  - Pulmão Esquerdo      : {esq_global} suspeitos")
    print(f"  - Pulmão Direito       : {dir_global} suspeitos")

    if dir_global > esq_global:
        lado_critico = "Direito"
    elif esq_global > dir_global:
        lado_critico = "Esquerdo"
    else:
        lado_critico = "Equilibrado"
    print(f"Maior Concentração       : Pulmão {lado_critico}")

    print("\n--- Auditoria por Processo (Faixas) ---")
    for rel in sorted(todos_relatorios, key=lambda r: r['rank']):
        print(f"Processo {rel['rank']:02d} | Linhas [{rel['linhas_inicio']:04d}-{rel['linhas_fim']:04d}] | "
              f"Suspeitos: {rel['suspeitos']:05d} | Máx: {rel['max_local']:03d} | Faixa: {rel['classificacao']}")

    if taxa_comprometida >= parametros['pct_critico']:
        diagnostico = "QUADRO CRÍTICO / ALTA CONCENTRAÇÃO DE ALTERAÇÕES"
    elif taxa_comprometida >= 1.0:
        diagnostico = "ATENÇÃO CLÍNICA"
    else:
        diagnostico = "SEM INDÍCIOS RELEVANTES"

    print(f"\nClassificação Geral da Radiografia: {diagnostico}")
    print("=" * 60 + "\n")