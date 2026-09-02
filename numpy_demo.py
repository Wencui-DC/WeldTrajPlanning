"""
NumPy 函数和数据结构学习Demo
============================
本文件涵盖NumPy的核心功能,包括:
- 数组创建方法
- 数组属性和形状操作
- 数学运算和统计函数
- 索引、切片和布尔索引
- 线性代数函数
- 随机数生成
- 广播机制
"""

import numpy as np

print("=" * 60)
print("NumPy 学习Demo")
print("=" * 60)

# ============================================================
# 第一部分:数组创建方法
# ============================================================
print("\n【第一部分】数组创建方法")
print("-" * 40)

# 1. 从Python列表创建数组
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"一维数组: {arr1d}")
print(f"二维数组:\n{arr2d}")
print(f"三维数组:\n{arr3d}")

# 2. 使用特殊函数创建数组
zeros = np.zeros((3, 4))          # 全零数组
ones = np.ones((2, 3))            # 全一数组
full = np.full((2, 3), 7)         # 填充特定值
eye = np.eye(3)                   # 单位矩阵
diag = np.diag([1, 2, 3, 4])      # 对角矩阵

print(f"\n全零数组 (3x4):\n{zeros}")
print(f"\n全一数组 (2x3):\n{ones}")
print(f"\n填充数组 (2x3, 值=7):\n{full}")
print(f"\n单位矩阵 (3x3):\n{eye}")
print(f"\n对角矩阵:\n{diag}")

# 3. 使用arange和linspace创建等差数组
arange_arr = np.arange(0, 10, 2)   # 起点0, 终点10(不含), 步长2
linspace_arr = np.linspace(0, 1, 5) # 0到1之间生成5个等间距点

print(f"\narange(0, 10, 2): {arange_arr}")
print(f"linspace(0, 1, 5): {linspace_arr}")

# 4. 其他创建方法
empty = np.empty((2, 3))          # 未初始化数组(值随机)
random_arr = np.random.rand(3, 3) # 0到1之间的随机数
identity = np.identity(4)         # 单位矩阵(eye的别名)

print(f"\n空数组 (2x3):\n{empty}")
print(f"随机数组 (3x3):\n{random_arr}")
print(f"单位矩阵 (4x4):\n{identity}")

# ============================================================
# 第二部分:数组属性和形状操作
# ============================================================
print("\n【第二部分】数组属性和形状操作")
print("-" * 40)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"原始数组:\n{arr}")

# 1. 数组属性
print(f"\n数组维度: {arr.ndim}")
print(f"数组形状: {arr.shape}")
print(f"数组大小(元素总数): {arr.size}")
print(f"数组数据类型: {arr.dtype}")
print(f"每个元素占字节数: {arr.itemsize}")
print(f"数组内存占用(字节): {arr.nbytes}")

# 2. reshape - 改变形状
reshaped = arr.reshape(4, 3)       # 改为4行3列
reshaped2 = arr.reshape(2, 6)      # 改为2行6列
reshaped_flat = arr.reshape(-1)    # 展平为一维(-1表示自动计算)
reshaped_expand = arr.reshape(2, 2, 3) # 改为3维

print(f"\nreshape(4, 3):\n{reshaped}")
print(f"\nreshape(2, 6):\n{reshaped2}")
print(f"\nreshape(-1) 展平: {reshaped_flat}")
print(f"\nreshape(2, 2, 3) 三维:\n{reshaped_expand}")

# 3. flatten和ravel - 展平数组
print(f"\nflatten(): {arr.flatten()}")  # 返回副本
print(f"ravel(): {arr.ravel()}")        # 返回视图(更高效)

# 4. transpose - 转置
transposed = arr.T
print(f"\n转置后:\n{transposed}")

# 5. 数组拼接
arr_a = np.array([[1, 2], [3, 4]])
arr_b = np.array([[5, 6], [7, 8]])

hstack = np.hstack((arr_a, arr_b))    # 水平拼接
vstack = np.vstack((arr_a, arr_b))    # 垂直拼接
concat_axis0 = np.concatenate((arr_a, arr_b), axis=0)  # 沿axis=0拼接
concat_axis1 = np.concatenate((arr_a, arr_b), axis=1)  # 沿axis=1拼接

print(f"\n水平拼接 hstack:\n{hstack}")
print(f"\n垂直拼接 vstack:\n{vstack}")
print(f"\nconcatenate axis=0:\n{concat_axis0}")
print(f"concatenate axis=1:\n{concat_axis1}")

# 6. 数组分割
split_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
h_split = np.hsplit(split_arr, 2)   # 水平分割为2份
v_split = np.vsplit(split_arr, 2)   # 垂直分割为2份

print(f"\n原始数组:\n{split_arr}")
print(f"\n水平分割为2份: 第一份\n{h_split[0]}")

# ============================================================
# 第三部分:数学运算和统计函数
# ============================================================
print("\n【第三部分】数学运算和统计函数")
print("-" * 40)

a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

# 1. 基本数学运算(逐元素)
print(f"数组a: {a}")
print(f"数组b: {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")       # 逐元素相乘
print(f"a / b = {a / b}")
print(f"a ** b = {a ** b}")     # 逐元素幂运算

# 2. 矩阵乘法
mat_a = np.array([[1, 2], [3, 4]])
mat_b = np.array([[5, 6], [7, 8]])

mat_mul = np.dot(mat_a, mat_b)        # 矩阵乘法
mat_mul2 = mat_a @ mat_b              # 矩阵乘法(另一种写法)

print(f"\n矩阵乘法 np.dot:\n{mat_mul}")
print(f"矩阵乘法 @:\n{mat_mul2}")

# 3. 统计函数
print(f"\n求和 sum: {np.sum(a)}")
print(f"均值 mean: {np.mean(a)}")
print(f"标准差 std: {np.std(a):.4f}")
print(f"方差 var: {np.var(a):.4f}")
print(f"最小值 min: {np.min(a)}")
print(f"最大值 max: {np.max(a)}")
print(f"最小值索引 argmin: {np.argmin(a)}")
print(f"最大值索引 argmax: {np.argmax(a)}")

# 沿特定轴计算
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n矩阵:\n{matrix}")
print(f"按行求和(沿axis=1): {np.sum(matrix, axis=1)}")
print(f"按列求和(沿axis=0): {np.sum(matrix, axis=0)}")
print(f"按行求均值: {np.mean(matrix, axis=1)}")
print(f"按列求均值: {np.mean(matrix, axis=0)}")

# 4. 其他常用数学函数
print(f"\nsin: {np.sin(np.array([0, np.pi/2, np.pi]))}")
print(f"cos: {np.cos(np.array([0, np.pi/2, np.pi]))}")
print(f"sqrt: {np.sqrt(np.array([1, 4, 9, 16]))}")
print(f"abs: {np.abs(np.array([-1, 2, -3, 4]))}")
print(f"exp: {np.exp(np.array([0, 1, 2]))}")
print(f"log: {np.log(np.array([1, np.e, np.e**2]))}")
print(f"round: {np.round(np.array([1.234, 5.678, 9.012]), decimals=2)}")

# ============================================================
# 第四部分:索引、切片和布尔索引
# ============================================================
print("\n【第四部分】索引、切片和布尔索引")
print("-" * 40)

arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120]])
print(f"原始数组:\n{arr}")

# 1. 基本索引和切片
print(f"\narr[0, 0] = {arr[0, 0]}")         # 单个元素
print(f"arr[1, :] = {arr[1, :]}")           # 第2行所有列
print(f"arr[:, 2] = {arr[:, 2]}")           # 所有行第3列
print(f"arr[0:2, 1:3]:\n{arr[0:2, 1:3]}")  # 切片

# 2. 花式索引(Fancy Indexing)
indices = [0, 2]                          # 行索引
print(f"\n行索引 [0,2]:\n{arr[indices, :]}")

row_idx = np.array([0, 1, 2])
col_idx = np.array([1, 2, 3])
print(f"对角元素: {arr[row_idx, col_idx]}")  # [arr[0,1], arr[1,2], arr[2,3]]

# 3. 布尔索引
print(f"\n大于50的元素: {arr[arr > 50]}")
print(f"等于60的元素: {arr[arr == 60]}")

# 4. 条件赋值
arr_copy = arr.copy()
arr_copy[arr_copy > 50] = 999              # 大于50的元素替换为999
print(f"\n条件赋值后(>50变为999):\n{arr_copy}")

# 5. np.where条件选择
result = np.where(arr > 50, arr, 0)        # 大于50保持原值,否则为0
print(f"\nnp.where选择:\n{result}")

# ============================================================
# 第五部分:线性代数函数
# ============================================================
print("\n【第五部分】线性代数函数")
print("-" * 40)

# 1. 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")
print(f"\n矩阵乘法(A @ B):\n{A @ B}")
print(f"转置(A.T):\n{A.T}")
inv_A = np.linalg.inv(A)
print(f"\n逆矩阵(np.linalg.inv(A)):\n{np.round(inv_A, 4)}")  # 保留4位小数

# 2. 行列式
det_A = np.linalg.det(A)
print(f"\n矩阵A的行列式: {det_A:.4f}")

# 3. 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 4. 线性方程组求解
# 求解 Ax = b
b = np.array([5, 7])
x = np.linalg.solve(A, b)
print(f"\n求解 Ax = b (b={b}):")
print(f"x = {x}")
print(f"验证 Ax = {A @ x}")

# 5. 奇异值分解(SVD)
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD分解:")
print(f"U:\n{U}")
print(f"奇异值: {S}")
print(f"Vt:\n{Vt}")

# 6. 范数
print(f"\nFrobenius范数: {np.linalg.norm(A, 'fro'):.4f}")
print(f"2-范数: {np.linalg.norm(A, 2):.4f}")

# ============================================================
# 第六部分:随机数生成
# ============================================================
print("\n【第六部分】随机数生成")
print("-" * 40)

# 设置随机种子(保证结果可复现)
np.random.seed(42)

# 1. 均匀分布随机数
uniform_arr = np.random.rand(3, 3)
print(f"均匀分布(0-1) 3x3:\n{uniform_arr}")

# 2. 正态分布随机数
normal_arr = np.random.randn(3, 3)
print(f"\n标准正态分布 3x3:\n{normal_arr}")

# 3. 指定范围的随机整数
randint_arr = np.random.randint(1, 100, size=(3, 3))
print(f"\n随机整数(1-100) 3x3:\n{randint_arr}")

# 4. 自定义分布
low, high = 5, 15
custom_uniform = np.random.uniform(low, high, size=(2, 3))
print(f"\n自定义均匀分布({low}-{high}) 2x3:\n{custom_uniform}")

mean, std = 0, 1
custom_normal = np.random.normal(mean, std, size=(2, 3))
print(f"\n自定义正态分布(mean={mean}, std={std}) 2x3:\n{custom_normal}")

# 5. 随机排列
arr_shuffle = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.random.shuffle(arr_shuffle)  # 原地打乱
print(f"\n打乱顺序: {arr_shuffle}")

# 6. 随机选择
arr_choice = np.array([10, 20, 30, 40, 50])
chosen = np.random.choice(arr_choice, size=3, replace=False)  # 无放回抽样
print(f"随机选择3个元素: {chosen}")

# 7. 随机种子的重要性
print("\n随机种子说明:")
print("设置np.random.seed(数字)可以使随机结果可复现")
print("相同种子+相同操作 = 相同结果")

# ============================================================
# 第七部分:广播机制
# ============================================================
print("\n【第七部分】广播机制")
print("-" * 40)

# 广播规则:
# 1. 如果两个数组维度不同,将shape较小的数组前面补1
# 2. 如果维度大小不同且不为1,报错
# 3. 大小为1的维度可以扩展到任意大小

# 1. 标量与数组运算(标量被广播)
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10

result = arr + scalar
print(f"数组 + 标量:\n{result}")

# 2. 一维数组与二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
arr_1d = np.array([10, 20, 30])             # shape: (3,)

result = arr_2d + arr_1d  # arr_1d被广播为(1,3)再与(2,3)相加
print(f"\n二维数组 + 一维数组:\n{result}")

# 3. 列向量与行向量
col_vec = np.array([[1], [2], [3]])   # shape: (3, 1)
row_vec = np.array([10, 20, 30])      # shape: (3,)

result = col_vec + row_vec  # 广播为(3,3)
print(f"\n列向量 + 行向量:\n{result}")

# 4. 三维与二维广播
arr_3d = np.random.rand(2, 3, 4)      # shape: (2, 3, 4)
arr_2d = np.random.rand(3, 4)         # shape: (3, 4)

result = arr_3d + arr_2d               # arr_2d被广播为(1,3,4)再相加
print(f"\n三维数组 + 二维数组结果形状: {result.shape}")

# 5. 广播实战示例 - 归一化
data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
print(f"\n原始数据:\n{data}")

# 归一化: (x - min) / (max - min)
data_min = data.min(axis=1, keepdims=True)  # shape: (3, 1)
data_max = data.max(axis=1, keepdims=True)  # shape: (3, 1)

normalized = (data - data_min) / (data_max - data_min)
print(f"按行归一化后:\n{normalized}")

# ============================================================
# 第八部分:实用技巧
# ============================================================
print("\n【第八部分】实用技巧")
print("-" * 40)

# 1. 类型转换
arr_float = np.array([1.5, 2.7, 3.9])
arr_int = arr_float.astype(int)  # 转为整数(截断)
print(f"浮点数转整数: {arr_float} -> {arr_int}")

# 2. 深拷贝vs浅拷贝
original = np.array([1, 2, 3])
view_copy = original.view()        # 浅拷贝(视图)
deep_copy = original.copy()        # 深拷贝(独立副本)

original[0] = 999
print(f"\n原始数组修改为999后:")
print(f"视图: {view_copy}")       # 会受影响
print(f"深拷贝: {deep_copy}")     # 不受影响

# 3. 保存和加载数组
save_arr = np.array([1, 2, 3, 4, 5])
np.save('test_array.npy', save_arr)     # 保存为.npy文件
loaded_arr = np.load('test_array.npy')  # 加载.npy文件
print(f"\n保存后加载: {loaded_arr}")

# 4. 排序
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_arr = np.sort(unsorted)
print(f"\n排序前: {unsorted}")
print(f"排序后: {sorted_arr}")
print(f"argsort(索引): {np.argsort(unsorted)}")  # 返回排序后的索引

# 5. 去重
duplicates = np.array([1, 2, 2, 3, 3, 3, 4])
unique = np.unique(duplicates)
print(f"\n去重前: {duplicates}")
print(f"去重后: {unique}")

# 6. 查找
print(f"\n查找元素3的位置: {np.where(duplicates == 3)}")

# ============================================================
# 第九部分:综合实战示例
# ============================================================
print("\n【第九部分】综合实战示例")
print("-" * 40)

# 示例1: 计算两点间的距离
point_a = np.array([1, 2, 3])
point_b = np.array([4, 5, 6])
distance = np.sqrt(np.sum((point_a - point_b)**2))
print(f"点A{point_a}和点B{point_b}之间的距离: {distance:.4f}")

# 示例2: 计算向量的夹角
vec_a = np.array([1, 0])
vec_b = np.array([0, 1])
cos_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
angle = np.arccos(cos_angle)
print(f"\n向量{vec_a}和{vec_b}的夹角: {np.degrees(angle):.4f}度")

# 示例3: 矩阵的秩
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(C)
print(f"\n矩阵C的秩: {rank}")

# 示例4: 最小二乘法拟合
np.random.seed(0)
x = np.linspace(0, 10, 20)
y = 2 * x + 3 + np.random.randn(20) * 2  # 带噪声的线性数据

# 使用numpy的最小二乘拟合
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"\n最小二乘拟合 y = {m:.4f}x + {c:.4f}")

# 示例5: 图像基本操作(模拟)
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
print(f"\n模拟图像形状: {image.shape}")
print(f"像素值范围: {image.min()} - {image.max()}")
print(f"平均亮度: {image.mean():.2f}")

# 示例6: 聚类中心计算
points = np.random.rand(100, 2)  # 100个二维点
centroid = points.mean(axis=0)
print(f"\n100个随机点的中心: {centroid}")

print("\n" + "=" * 60)
print("NumPy学习Demo完成!")
print("=" * 60)
