import re

import sympy as sp


def split_formula(formula):
    formula = formula.replace(' ', '')
    terms = re.findall(r'([+-]?\d*x?\^?\d*)B\d+', formula)
    return terms


def evaluate_symbolic_expression(formula, x_value):
    x = sp.Symbol('x')

    expression = sp.sympify(formula)
    result = expression.subs(x, x_value)

    return result


formula = "1B0 + xB1 - 2*x^2B2"
terms = split_formula(formula)
print(terms)

x_values = [2, 3, 4]

values = []
for x in x_values:
    qwe = []
    for term in terms:
        result = evaluate_symbolic_expression(term, x)
        qwe.append(result)
        print(f"For x = {x}, the formula evaluates to: {result}")
    values.append(qwe)

print(values)

from Tools import  transpose_matrix
print(transpose_matrix(values))