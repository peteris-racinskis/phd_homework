import numpy as np
from math import prod, log2
from argparse import ArgumentParser


class OpCounter():

    def __init__(self, m=1, print=False):
        self.incr_mul = 0
        self.incr_add = 0
        self.m = m
        self.print = print

    def mul_cnt(self, a: np.ndarray, b: np.ndarray):
        assert(a.shape == b.shape)
        assert(a.shape[0] == a.shape[1])
        if a.shape[0] != self.m:
            return strassen_mul(a, b, self)
        else:
            self.incr_mul += a.shape[0] ** 3
            self.incr_add += (a.shape[0] ** 2) * (a.shape[0] - 1)
            if self.print:
                print(f"+ {a.shape[0] ** 3} multiplications")
                print(f"+ {(a.shape[0] ** 2) * (a.shape[0] - 1)} additions")
            result = a @ b
            return result

    def add_cnt(self, a: np.ndarray, b: np.ndarray):
        assert(a.shape == b.shape)
        assert(a.shape[0] == a.shape[1])
        self.incr_add += prod(a.shape)
        result = a + b
        if self.print:
            print(f"+ {prod(a.shape)} additions")
        return result


def strassen_mul(a, b, c):

    h,w = a.shape

    a_11, b_11 = a[:h//2,:w//2], b[:h//2,:w//2]
    a_12, b_12 = a[:h//2,w//2:], b[:h//2,w//2:]
    a_21, b_21 = a[h//2:,:w//2], b[h//2:,:w//2]
    a_22, b_22 = a[h//2:,w//2:], b[h//2:,w//2:]

    S_1 = c.mul_cnt( c.add_cnt(a_11, a_22), c.add_cnt(b_11,b_22) )
    S_2 = c.mul_cnt( c.add_cnt(a_21, a_22), b_11 )
    S_3 = c.mul_cnt( a_11, c.add_cnt(b_12, -b_22) )
    S_4 = c.mul_cnt( a_22, c.add_cnt(-b_11, b_21) )
    S_5 = c.mul_cnt( c.add_cnt(a_11,a_12), b_22 )
    S_6 = c.mul_cnt( c.add_cnt(-a_11,a_21), c.add_cnt(b_11,b_12) )
    S_7 = c.mul_cnt( c.add_cnt(a_12,-a_22), c.add_cnt(b_21,b_22) )

    C_11 = c.add_cnt( c.add_cnt(S_1,S_4), c.add_cnt(-S_5,S_7) )
    C_21 = c.add_cnt( S_2, S_4 )
    C_12 = c.add_cnt( S_3, S_5 )
    C_22 = c.add_cnt( c.add_cnt(S_1, S_3), c.add_cnt(-S_2, S_6) )

    top = np.concatenate([C_11,C_12], axis=-1)
    bottom = np.concatenate([C_21,C_22], axis=-1)

    result = np.concatenate([top,bottom], axis=0)

    return result

def random_mul(m, k, _print=True):

    n = m * (2**k)
    a = np.random.rand(n,n)
    b = np.random.rand(n,n)
    cnt = OpCounter(m, _print)
    result = strassen_mul(a,b,cnt)
    assert(np.allclose(result, a @ b))
    
    strassen_total = cnt.incr_mul + cnt.incr_add
    add_total = n**2 * (n-1)
    mul_total = n**3

    print(f"n = {n} elements = {n**2} muls = {cnt.incr_mul} adds = {cnt.incr_add}")
    print(f"Reference n^3 = {mul_total} n^2(n-1) = {add_total}")
    print(f"Strassen advantage mul {mul_total - cnt.incr_mul} add {add_total - cnt.incr_add} total {add_total + mul_total - strassen_total}")

def id_pad(m, new_n):
    new = np.identity(new_n)
    new[:m.shape[0], :m.shape[1]] = m
    return new

def embed_mul(n, _print=True):

    k = int(max(log2(n) - 4, 1))
    m = int(n * ( 2 ** -k )) + 1
    nn = m * (2**k)
    a = np.random.rand(n,n)
    b = np.random.rand(n,n)
    cnt = OpCounter(m, _print)

    result = strassen_mul(id_pad(a,nn),id_pad(b,nn),cnt)
    assert(np.allclose(result[:n,:n], a @ b))
    
    strassen_total = cnt.incr_mul + cnt.incr_add
    add_total = n**2 * (n-1)
    mul_total = n**3

    print(f"n = {n} elements = {n**2} m = {m} k = {k} nn = {nn} muls = {cnt.incr_mul} adds = {cnt.incr_add}")
    print(f"Reference n^3 = {mul_total} n^2(n-1) = {add_total}")
    print(f"Strassen advantage mul {mul_total - cnt.incr_mul} add {add_total - cnt.incr_add} total {add_total + mul_total - strassen_total}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", default=1, type=int, help="n = M * 2^(k); default = 1")
    parser.add_argument("-k", default=2, type=int, help="n = m * 2^(K); default = 2")
    parser.add_argument("-n", default=-1, required=False, type=int, help="for calc using embedding in next largest m * 2^(k); see strassen paper")
    parser.add_argument("-p", default=False, required=False, action="store_true", help="print irreducible operation counts at every step")
    args = parser.parse_args()

    if not args.n == -1:
        embed_mul(args.n, args.p)
    else:
        random_mul(args.m, args.k, args.p)



