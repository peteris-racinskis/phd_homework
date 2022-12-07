import numpy as np
from math import log2

def id_pad(m, new_n):
    new = np.identity(new_n)
    new[:m.shape[0], :m.shape[1]] = m
    return new

a = np.random.rand(5,5)
b = np.random.rand(5,5)

p_a = id_pad(a, 6)
p_b = id_pad(b, 6)

x = a @ b
p_x = p_a @ p_b

print(x)
print(p_x)

for n in range(1,100):
    k = int(max(log2(n) - 4, 0))
    m = int(n * ( 2 ** -k )) + 1
    print(f"n = {n} k = {k} m = {m}")