# 第三章
import numpy as np
from numpy.linalg import lstsq  # 最小二乘法

# 定义基函数，这里P0, P1, P2, P3分别对应1, x, x^2, x^3
def basis_functions(x):
    return np.vstack([np.ones_like(x), x, x**2, x**3]).T

# 定义目标函数f(x) = e^x
def target_function(x):
    return np.exp(x)

# 定义区间[-1, 1]上的点，用于计算逼近多项式
x_points = np.linspace(-1, 1, 100)

# 计算基函数和目标函数在这些点上的值
A = basis_functions(x_points)
y = target_function(x_points)

# 使用最小二乘法求解逼近多项式的系数
coefficients, _, _, _ = lstsq(A, y, rcond=None)

# 打印逼近多项式的系数
print("逼近多项式的系数为：", coefficients)

# 定义逼近多项式函数
def approximated_polynomial(x):
    return coefficients[0] + coefficients[1] * x + coefficients[2] * x**2 + coefficients[3] * x**3

# 可以选择验证逼近多项式在区间[-1, 1]上的表现
x_test = np.linspace(-1, 1, 10)
y_test = approximated_polynomial(x_test)
print("测试点上的逼近多项式值：", y_test)
