import random
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

dentro = 0
inicio = time.time()
for _ in range(N):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1.0:
        dentro += 1

pi_estimado = 4.0 * dentro / N
fim = time.time()

print(f"PI aproximado: {pi_estimado:.6f}")
print(f"Tempo sequencial: {(fim - inicio) * 1000:.2f} ms")