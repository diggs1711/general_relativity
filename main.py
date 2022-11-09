from sympy import symbols, Function, Pow, MutableDenseNDimArray, diff, init_printing, pprint, simplify
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
                            (diff(metric[d, b], symbols[c]) + diff(metric[d, c],
                             symbols[b]) - diff(metric[b, c], symbols[d]))
                values[a, b, c] = val
    return values


def riemann_tensor(christoffel_symbols):
    values = MutableDenseNDimArray.zeros(
        COORDINATES, COORDINATES, COORDINATES, COORDINATES)
    for d in range(0, COORDINATES):
        for a in range(0, COORDINATES):
            for b in range(0, COORDINATES):
                for c in range(0, COORDINATES):
                    val1 = 0
                    for e in range(0, COORDINATES):
                        val1 += christoffel_symbols[e, a,
                                                    c]*christoffel_symbols[d, e, b]
                    val2 = 0
                    for e in range(0, COORDINATES):
                        val2 += christoffel_symbols[e, a,
                                                    b]*christoffel_symbols[d, e, c]

                    values[d, a, b, c] += diff(christoffel_symbols[d, a, c], symbols[b]) - diff(
                        christoffel_symbols[d, a, b], symbols[c]) + val1 - val2

    return values


def ricci_tensor(riemann_tensor):
    values = MutableDenseNDimArray.zeros(COORDINATES, COORDINATES)
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            val = 0
            for c in range(0, COORDINATES):
                val += riemann_tensor[c, a, c, b]
            values[a, b] = val
    return values


def ricci_scalar(ricci_tensor, metric):
    result = 0
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            if metric[a, b] != 0:
                result += (1/metric[a, b])*(ricci_tensor[a, b])
    return result


def einstein_tensor(ricci_tensor, ricci_scalar, metric):
    result = MutableDenseNDimArray.zeros(COORDINATES, COORDINATES)
    for a in range(0, COORDINATES):
        for b in range(0, COORDINATES):
            result[a, b] = ricci_tensor[a, b] - \
                0.5*(ricci_scalar*(metric[a, b]))
    return result


metric = MutableDenseNDimArray.zeros(COORDINATES, COORDINATES)

t = symbols[0]
metric[0, 0] = Pow(c, 2)
metric[1, 1] = -Pow(a(t), 2)
metric[2, 2] = -Pow(a(t), 2)
metric[3, 3] = -Pow(a(t), 2)


chris_syms = christoffel_symbols(metric)
# pprint(chris_syms)
riemann = riemann_tensor(chris_syms)
# pprint(riemann)
r_tensor = ricci_tensor(riemann)
r_scalar = ricci_scalar(r_tensor, metric)
# pprint(simplify(r_scalar))
einstein = einstein_tensor(r_tensor, r_scalar, metric)
pprint(simplify(einstein[0, 0]))
