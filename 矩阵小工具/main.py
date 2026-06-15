# Test code for IEEE course final project
# Fan Cheng, 2024
import time
import minimatrix_demo as mm
import random


def test_all():
    """测试所有功能"""
    seed_value = int(time.time() * 1000)  # 乘以1000获取毫秒级精度
    random.seed(seed_value)  # 保证随机测试可复现
    print("=" * 60)
    print("MiniMatrix 测试套件")
    print("=" * 60)

    # 1. 测试基本创建
    print("\n1. 测试基本创建")
    print("-" * 40)

    # 从列表创建
    print("从列表创建矩阵:")
    mat1 = mm.Matrix(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"mat1 = \n{mat1}")
    print(f"mat1.shape() = {mat1.shape()}")

    # 使用dim创建
    print("\n使用dim创建零矩阵:")
    mat2 = mm.Matrix(dim=(2, 3), init_value=0)
    print(f"mat2 = \n{mat2}")

    # 使用dim创建特定值矩阵
    print("\n使用dim创建全1矩阵:")
    mat3 = mm.Matrix(dim=(3, 2), init_value=1)
    print(f"mat3 = \n{mat3}")

    # 2. 测试索引功能
    print("\n\n2. 测试索引功能")
    print("-" * 40)

    test_mat = mm.Matrix(data=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    print("测试矩阵:")
    print(test_mat)

    # 单个元素访问
    print(f"\ntest_mat[0, 0] = {test_mat[0, 0]}")
    print(f"test_mat[1, 2] = {test_mat[1, 2]}")
    print(f"test_mat[-1, -1] = {test_mat[-1, -1]}")
    print(f"test_mat[-2, -3] = {test_mat[-2, -3]}")

    # 切片访问
    print("\n切片访问:")
    print("test_mat[0:2, 1:3] =")
    print(test_mat[0:2, 1:3])

    print("\ntest_mat[:, :2] =")
    print(test_mat[:, :2])

    print("\ntest_mat[1:, :] =")
    print(test_mat[1:, :])

    print("\ntest_mat[::2, ::2] =")
    print(test_mat[::2, ::2])

    # 3. 测试赋值功能
    print("\n\n3. 测试赋值功能")
    print("-" * 40)

    assign_mat = mm.Matrix(data=[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ])
    print("原始矩阵:")
    print(assign_mat)

    # 单个元素赋值
    assign_mat[1, 2] = 99.0
    print("\n赋值后 assign_mat[1, 2] = 99:")
    print(assign_mat)

    assign_mat[-1, -1] = 88.0
    print("\n赋值后 assign_mat[-1, -1] = 88:")
    print(assign_mat)

    # 切片赋值
    assign_mat[1:, 2:] = mm.Matrix(data=[[100, 200], [300, 400]])
    print("\n切片赋值后:")
    print(assign_mat)

    # 4. 测试矩阵运算
    print("\n\n4. 测试矩阵运算")
    print("-" * 40)

    A = mm.Matrix(data=[[1, 2], [3, 4]])
    B = mm.Matrix(data=[[5, 6], [7, 8]])

    print(f"A = \n{A}")
    print(f"B = \n{B}")

    # 矩阵乘法
    print(f"\nA.dot(B) = \n{A.dot(B)}")

    # 转置
    print(f"\nA.T() = \n{A.T()}")

    # 元素对应运算
    print(f"\nA + B = \n{A + B}")
    print(f"\nA - B = \n{A - B}")
    print(f"\nA * B (对应元素相乘) = \n{A * B}")

    # 幂运算
    print(f"\nA ** 2 = \n{A ** 2}")
    print(f"\nA ** 3 = \n{A ** 3}")

    # 行列式、逆、秩、Gauss 消元
    print("\n附加: 行列式 / 逆 / 秩 / Gauss 消元")
    sq = mm.Matrix(data=[[1, 2], [3, 5]])
    print(f"sq = \n{sq}")
    print(f"sq.det() = {sq.det()}")
    print(f"sq.inverse() = \n{sq.inverse()}")
    sing = mm.Matrix(data=[[1, 2], [2, 4]])
    print(f"sing.rank() (应为1) = {sing.rank()}")
    print(f"sq.gauss_elimination() = \n{sq.gauss_elimination()}")

    # 5. 测试求和
    print("\n\n5. 测试求和")
    print("-" * 40)

    sum_mat = mm.Matrix(data=[[1, 2, 3], [4, 5, 6]])
    print(f"sum_mat = \n{sum_mat}")
    print(f"sum_mat.sum() = \n{sum_mat.sum()}")
    print(f"sum_mat.sum(axis=0) = \n{sum_mat.sum(axis=0)}")
    print(f"sum_mat.sum(axis=1) = \n{sum_mat.sum(axis=1)}")

    # 6. 测试reshape
    print("\n\n6. 测试reshape")
    print("-" * 40)

    orig = mm.Matrix(data=[[1, 2, 3, 4], [5, 6, 7, 8]])
    print(f"原始矩阵: \n{orig}")
    print(f"reshape为(4, 2): \n{orig.reshape((4, 2))}")
    print(f"reshape为(1, 8): \n{orig.reshape((1, 8))}")
    print(f"reshape为(8, 1): \n{orig.reshape((8, 1))}")

    # 7. 测试单位矩阵和特殊矩阵函数
    print("\n\n7. 测试特殊矩阵函数")
    print("-" * 40)

    print(f"单位矩阵 I(3): \n{mm.I(3)}")
    print(f"单位矩阵 I(4): \n{mm.I(4)}")

    print(f"\n全零矩阵 zeros((2, 3)): \n{mm.zeros((2, 3))}")
    print(f"全一矩阵 ones((3, 2)): \n{mm.ones((3, 2))}")

    print(f"\narange(0, 10, 2): \n{mm.arange(0, 10, 2)}")

    # 8. 测试Kronecker积
    print("\n\n8. 测试Kronecker积")
    print("-" * 40)

    K1 = mm.Matrix(data=[[1, 2], [3, 4]])
    K2 = mm.Matrix(data=[[0, 5], [6, 7]])
    print(f"K1 = \n{K1}")
    print(f"K2 = \n{K2}")
    print(f"Kronecker积 K1 ⊗ K2: \n{K1.Kronecker_product(K2)}")

    # 9. 测试拼接
    print("\n\n9. 测试拼接")
    print("-" * 40)

    M1 = mm.Matrix(data=[[1, 2], [3, 4]])
    M2 = mm.Matrix(data=[[5, 6], [7, 8]])
    M3 = mm.Matrix(data=[[9, 10], [11, 12]])

    print(f"M1 = \n{M1}")
    print(f"M2 = \n{M2}")
    print(f"M3 = \n{M3}")

    print(f"\n沿行拼接 concatenate([M1, M2, M3], axis=0):")
    print(mm.concatenate([M1, M2, M3], axis=0))

    print(f"\n沿列拼接 concatenate([M1, M2, M3], axis=1):")
    print(mm.concatenate([M1, M2, M3], axis=1))

    # 10. 辅助创建函数
    print("\n\n10. 辅助创建函数测试")
    print("-" * 40)
    print(f"narray((2,2), init_value=9): \n{mm.narray((2,2), init_value=9)}")
    base = mm.Matrix(data=[[7, 8], [9, 10]])
    print(f"zeros_like(base): \n{mm.zeros_like(base)}")
    print(f"ones_like(base): \n{mm.ones_like(base)}")
    print(f"nrandom_like(base) 形状: {mm.nrandom_like(base).shape()}")

    # 11. 测试vectorize
    print("\n\n11. 测试vectorize")
    print("-" * 40)

    V = mm.Matrix(data=[[-1, 2], [3, -4]])
    print(f"原始矩阵 V = \n{V}")

    @mm.vectorize
    def square(x):
        return x ** 2

    print(f"\n平方函数 square(V): \n{square(V)}")

    @mm.vectorize
    def my_abs(x):
        return -x if x < 0 else x

    print(f"\n绝对值函数 my_abs(V): \n{my_abs(V)}")

    # 数乘与长度
    print("\n数乘与长度测试")
    print(f"num_mul(A, 10) = \n{mm.num_mul(A, 10)}")
    print(f"len(A) = {len(A)}")

    # 12. 测试copy和相等判断
    print("\n\n12. 测试copy功能")
    print("-" * 40)

    original = mm.Matrix(data=[[1, 2], [3, 4]])
    copied = original.copy()
    print(f"original = \n{original}")
    print(f"copied = \n{copied}")

    # 修改副本，检查原矩阵是否不变
    copied[0, 0] = 99.0
    print(f"\n修改copied[0,0]=99后:")
    print(f"original = \n{original}")
    print(f"copied = \n{copied}")

    # 13. 测试异常处理
    print("\n\n13. 测试异常处理")
    print("-" * 40)

    try:
        print("测试非法创建: Matrix()")
        bad = mm.Matrix()
    except Exception as e:
        print(f"捕获异常: {type(e).__name__}: {e}")

    try:
        print("\n测试非法reshape:")
        mat = mm.Matrix(dim=(2, 3))
        mat.reshape((4, 4))
    except Exception as e:
        print(f"捕获异常: {type(e).__name__}: {e}")

    try:
        print("\n测试形状不匹配的矩阵乘法:")
        m1 = mm.Matrix(dim=(2, 3))
        m2 = mm.Matrix(dim=(4, 5))
        m1.dot(m2)
    except Exception as e:
        print(f"捕获异常: {type(e).__name__}: {e}")


    # 14.测试库应用：最小二乘法估计
    print("\n\n14. 测试库应用：最小二乘法估计")
    print("-" * 40)

    X = mm.nrandom((8, 5))
    print(X)
    w = mm.nrandom((5, 1))
    temp = [random.randint(1, 100) for _ in range(8)]
    s = sum(temp) / len(temp)
    e = mm.Matrix([[x - s for x in temp]]).T()
    Y = X.dot(w) + e
    ww = (X.T().dot(X)).inverse().dot(X.T()).dot(Y)



    print("\n" + "=" * 60)
    print("测试完成！")
    if isinstance(ww, tuple):
        print(ww[2])
    print(f"w = \n{w}")
    print(f"ww = \n{ww}")
    print(f"两者相减，结果为：\n{ww-w}")


def performance_test():
    """性能测试"""
    print("\n\n性能测试")
    print("-" * 40)

    import time

    # 测试矩阵乘法性能
    n = 100
    print(f"创建两个 {n}x{n} 的随机矩阵...")
    A = mm.nrandom((n, n))
    B = mm.nrandom((n, n))

    start = time.time()
    C = A.dot(B)
    end = time.time()

    print(f"{n}x{n} 矩阵乘法耗时: {end - start:.4f} 秒")

    # 测试转置性能
    start = time.time()
    T = A.T()
    end = time.time()
    print(f"{n}x{n} 矩阵转置耗时: {end - start:.4f} 秒")


if __name__ == "__main__":
    # 运行所有测试
    test_all()

    # 可选：运行性能测试（对于较大的矩阵）
    performance_test()


