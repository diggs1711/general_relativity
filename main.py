from sympy import symbols, integrate, Symbol, Function, Pow, zeros, MutableDenseNDimArray, Derivative, diff, init_printing, pprint
from sympy.physics.units.quantities import Quantity

init_printing()

symbols = symbols('t x y z')
c = Quantity('c')
a = Function('a')

'''Constants'''
COORDINATES = 4


def christoffel_symbols(metric: MutableDenseNDimArray):
    values = MutableDenseNDimArray.zeros(COORDINATES, COORDINATES, COORDINATES)
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            for c in range(0, COORDINATES):
                val = 0
                for d in range(0, COORDINATES):
                    if (metric[a, d] != 0):
                        val += 0.5*(1/metric[a, d]) * \
                            (diff(metric[d, b], symbols[c]).doit() + diff(metric[d, c],
                             symbols[b]).doit() - diff(metric[b, c], symbols[d]).doit())
                values[a, b, c] = val
    return values


def riemann_tensor(christoffel_symbols):
    values = MutableDenseNDimArray.zeros(
        COORDINATES, COORDINATES, COORDINATES, COORDINATES)
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            for c in range(0, COORDINATES):
                for d in range(0, COORDINATES):
                    val = 0
                    for e in range(0, COORDINATES):
                        val += diff(christoffel_symbols[d, a, c], symbols[b]) - diff(
                            christoffel_symbols[d, a, b], symbols[b]) + christoffel_symbols[e, a, c]*christoffel_symbols[d, e, b] - christoffel_symbols[e, a, b]*christoffel_symbols[d, e, c]
                    values[d, a, b, c] = val
    return values


def ricci_tensor(riemann_tensor):
    values = MutableDenseNDimArray(COORDINATES, COORDINATES)
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            val = 0
            for c in range(0, COORDINATES):
                val += riemann_tensor[c, a, b, c]
            values[a, b] = val
    return values


metric = MutableDenseNDimArray.zeros(COORDINATES, COORDINATES)

t = symbols[0]
metric[0, 0] = c
metric[1, 1] = -Pow(a(t), 2)
metric[2, 2] = -Pow(a(t), 2)
metric[3, 3] = -Pow(a(t), 2)


chris_syms = christoffel_symbols(metric)
riemann = riemann_tensor(chris_syms)
pprint(riemann)
ricci = ricci_tensor(riemann)
pprint(ricci)
