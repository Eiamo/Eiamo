from scipy.integrate import quad
import numpy as np
from fractions import Fraction

# 定义被积函数
def integrand0(x):
    return 1

def integrand1(x):
    return x

def integrand2(x):
    return x**2

def integrand3(x):
    return x**3

def integrand4(x):
    return x**4

# 计算定积分
results = [quad(integrand, 0, 3)[0] for integrand in [integrand0, integrand1, integrand2, integrand3, integrand4]]

n = 4
x_points = np.arange(n)  # [0, 1, 2, 3]
L = np.zeros((n, n))

# 计算拉格朗日基函数的系数
for i in range(n):
    for j in range(n):
        if i != j:
            L[i, j] = np.prod([(x_points[i] - x_points[k]) / (x_points[j] - x_points[k]) for k in range(n) if k != i and k != j])
        else:
            L[i, j] = 1

# 解线性方程组找到系数
coefficients = np.linalg.solve(L, results[:n])  
fraction = [Fraction(c).limit() for c in coefficients]

# 结果
print(f"积分近似为：{fraction[0]}*f(0) + {fraction[1]}*f(1) + {fraction[2]}*f(2) + {fraction[3]}*f(3)")

