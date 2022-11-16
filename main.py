from sympy import (Function, MutableDenseNDimArray, Pow, diff, init_printing,
                   pprint, simplify, symbols)
from sympy.physics.units.quantities import Quantity

init_printing()

'''Constants'''
COORDINATES = symbols('t x y z')
c = Quantity('c')
a = Function('a')
r = Quantity('r')
NUM_COORDINATES = len(COORDINATES)


def christoffel_symbols(metric: MutableDenseNDimArray):
    values = MutableDenseNDimArray.zeros(
        NUM_COORDINATES, NUM_COORDINATES, NUM_COORDINATES)
    for a in range(0, NUM_COORDINATES):
        for b in range(0, NUM_COORDINATES):
            for c in range(0, NUM_COORDINATES):
                val = 0
                for d in range(0, NUM_COORDINATES):
                    if (metric[a, d] != 0):
                        val += 0.5*(1/metric[a, d]) * \
                            (diff(metric[d, b], COORDINATES[c]) + diff(metric[d, c],
                             COORDINATES[b]) - diff(metric[b, c], COORDINATES[d]))
                values[a, b, c] = val
    return simplify(values)


def riemann_tensor(christoffel_symbols):
    values = MutableDenseNDimArray.zeros(
        NUM_COORDINATES, NUM_COORDINATES, NUM_COORDINATES, NUM_COORDINATES)
    for d in range(0, NUM_COORDINATES):
        for a in range(0, NUM_COORDINATES):
            for b in range(0, NUM_COORDINATES):
                for c in range(0, NUM_COORDINATES):
                    val1 = 0
                    val2 = 0
                    for e in range(0, NUM_COORDINATES):
                        val1 += christoffel_symbols[e, a,
                                                    c]*christoffel_symbols[d, e, b]
                        val2 += christoffel_symbols[e, a,
                                                    b]*christoffel_symbols[d, e, c]
                    values[d, a, b, c] += diff(christoffel_symbols[d, a, c], COORDINATES[b]) - diff(
                        christoffel_symbols[d, a, b], COORDINATES[c]) + val1 - val2
    return simplify(values)


def ricci_tensor(riemann_tensor):
    values = MutableDenseNDimArray.zeros(NUM_COORDINATES, NUM_COORDINATES)
    for a in range(0, NUM_COORDINATES):
        for b in range(0, NUM_COORDINATES):
            val = 0
            for c in range(0, NUM_COORDINATES):
                val += riemann_tensor[c, a, b, c]
            values[a, b] = val
    return simplify(values)


def ricci_scalar(ricci_tensor, metric):
    result = 0
    for a in range(0, NUM_COORDINATES):
        for b in range(0, NUM_COORDINATES):
            if metric[a, b] != 0:
                result += (1/metric[a, b])*(ricci_tensor[a, b])
    return simplify(result)


def einstein_tensor(ricci_tensor, ricci_scalar, metric):
    result = MutableDenseNDimArray.zeros(NUM_COORDINATES, NUM_COORDINATES)
    for a in range(0, NUM_COORDINATES):
        for b in range(0, NUM_COORDINATES):
            result[a, b] = ricci_tensor[a, b] - \
                0.5*(ricci_scalar*(metric[a, b]))
    return simplify(result)


def calculate_metric():
    t = COORDINATES[0]
    metric = MutableDenseNDimArray.zeros(NUM_COORDINATES, NUM_COORDINATES)
    metric[0, 0] = Pow(c, 2)
    metric[1, 1] = -Pow(a(t), 2)
    metric[2, 2] = -Pow(a(t), 2)
    metric[3, 3] = -Pow(a(t), 2)
    return metric


if __name__ == '__main__':
    metric = calculate_metric()

    chris_syms = christoffel_symbols(metric)
    riemann = riemann_tensor(chris_syms)
    r_tensor = ricci_tensor(riemann)
    r_scalar = ricci_scalar(r_tensor, metric)
    e_tensor = einstein_tensor(r_tensor, r_scalar, metric)

    pprint(chris_syms)
    pprint(r_tensor)
    pprint(e_tensor)
