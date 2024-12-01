# 第二章
import numpy as np

# 定义牛顿插值函数
def newton_interpolation(x_points, y_points, x):
    n = len(x_points) - 1
    diff_table = np.zeros((n + 1, n + 1))
    diff_table[:, 0] = y_points  # 初始化差商表的第一列为函数值

    # 计算差商表
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_points[i + j] - x_points[i])

    # 计算插值多项式值
    result = diff_table[0, 0]
    for j in range(1, n + 1):
        term = diff_table[0, j]
        for k in range(j):
            term *= (x - x_points[k])
        result += term

    return result

# 选择插值节点，这里假设使用了x=0, π/6, π/4, π/3（对应sin0°, sin30°, sin45°, sin60°）
x_points = [0, np.pi / 6, np.pi / 4, np.pi / 3]
y_points = [np.sin(x) for x in x_points]

# 计算sin50°
x = np.deg2rad(50)  # 将角度转换为弧度
sin_50 = newton_interpolation(x_points, y_points, x)

# 输出结果
print(f"使用牛顿插值法计算的sin50°值为：{sin_50:.6f}")